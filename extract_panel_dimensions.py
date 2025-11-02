#!/usr/bin/env python3
"""
Extract per-panel 2D dimensions from dataset pattern specifications and
export single-panel images organized by the requested folder structure.

Output structure:
  images/shape/<garment_type>/<panel_name>/panel_<WIDTH>x<HEIGHT>_<ID>.<ext>

Where WIDTH and HEIGHT are in centimeters, rounded to nearest integer.
This creates garment pattern panels that will be fitted onto cloth materials.

Sampling:
  By default, the script samples up to 10% of datapoints per cloth-type
  folder, capped at 100 and floored at 10 (if available). You can override
  via CLI flags.

Usage examples:
  python "utility scripts/extract_panel_dimensions.py" \
    --data-root ./data \
    --output-root ./images/cloth

  python "utility scripts/extract_panel_dimensions.py" \
    --data-root ./data \
    --output-root ./images/cloth \
    --per-cloth-max 50 --sample-fraction 0.05 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# Optional dependencies for SVG rendering and PNG conversion
try:
    import svgwrite  # type: ignore
except Exception:
    svgwrite = None  # type: ignore

try:
    from svglib import svglib  # type: ignore
    from reportlab.graphics import renderPM  # type: ignore
except Exception:
    svglib = None  # type: ignore
    renderPM = None  # type: ignore

try:
    import cairosvg  # type: ignore
except Exception:
    cairosvg = None  # type: ignore


@dataclass
class Panel:
    name: str
    vertices: List[List[float]]  # list of [x, y]
    edges: List[Dict]

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def width_cm(self) -> float:
        x0, _, x1, _ = self.bbox
        return x1 - x0

    @property
    def height_cm(self) -> float:
        _, y0, _, y1 = self.bbox
        return y1 - y0


@dataclass
class Spec:
    datapoint_id: str
    garment_type: str
    panels: List[Panel]


def read_spec(spec_path: Path, garment_type: str) -> Spec:
    with spec_path.open("r") as f:
        spec = json.load(f)

    pattern = spec["pattern"]
    datapoint_id = Path(spec_path).parent.name
    panels: List[Panel] = []
    for panel_name, panel_data in pattern["panels"].items():
        panels.append(
            Panel(
                name=panel_name,
                vertices=panel_data["vertices"],
                edges=panel_data["edges"],
            )
        )

    return Spec(datapoint_id=datapoint_id, garment_type=garment_type, panels=panels)


def _control_to_abs_coord(
    start: Tuple[float, float], end: Tuple[float, float], control_scale: List[float]
) -> Tuple[float, float]:
    """
    Convert relative curvature control point [u, v] to absolute coordinates.
    Matches logic in packages/pattern/core.py::_control_to_abs_coord, adapted for 2D.
    start, end are 2D points; control_scale = [t, k] where t along edge, k along edge-perpendicular.
    """
    sx, sy = start
    ex, ey = end
    edge_x = ex - sx
    edge_y = ey - sy
    # perpendicular vector
    edge_perp_x = -edge_y
    edge_perp_y = edge_x

    control_start_x = sx + control_scale[0] * edge_x
    control_start_y = sy + control_scale[0] * edge_y
    control_x = control_start_x + control_scale[1] * edge_perp_x
    control_y = control_start_y + control_scale[1] * edge_perp_y
    return (control_x, control_y)


def draw_panel_svg(
    panel: Panel, out_svg: Path, scale_px_per_cm: float = 3.0, padding_px: int = 40
) -> None:
    if svgwrite is None:
        raise RuntimeError("svgwrite is not installed; cannot render SVG")

    # Convert vertices to a local coordinate system for drawing: flip Y down for image coordinates
    # and translate to (0,0) based on bbox, then scale from cm to pixels.
    x0, y0, x1, y1 = panel.bbox
    width_px = int(math.ceil((x1 - x0) * scale_px_per_cm)) + 2 * padding_px
    height_px = int(math.ceil((y1 - y0) * scale_px_per_cm)) + 2 * padding_px

    dwg = svgwrite.Drawing(
        str(out_svg), profile="full", size=(f"{width_px}px", f"{height_px}px")
    )

    def to_px(pt: Tuple[float, float]) -> Tuple[float, float]:
        # shift to bbox origin, flip Y for SVG downward axis, apply padding and scale
        x = (pt[0] - x0) * scale_px_per_cm + padding_px
        y = (y1 - pt[1]) * scale_px_per_cm + padding_px  # invert Y
        return (x, y)

    # Build a path from edges in given order; assumes a closed loop
    verts = panel.vertices
    edges = panel.edges
    if not edges:
        return

    # Start point: first edge's first vertex
    start_idx = edges[0]["endpoints"][0]
    start = to_px((verts[start_idx][0], verts[start_idx][1]))
    path_cmds: List = ["M", start[0], start[1]]

    for edge in edges:
        s_idx, e_idx = edge["endpoints"]
        e = to_px((verts[e_idx][0], verts[e_idx][1]))
        if "curvature" in edge:
            # Quadratic Bezier control point in absolute (computed from relative control)
            control_abs = _control_to_abs_coord(
                (verts[s_idx][0], verts[s_idx][1]),
                (verts[e_idx][0], verts[e_idx][1]),
                edge["curvature"],
            )  # type: ignore
            c = to_px(control_abs)
            path_cmds += ["Q", c[0], c[1], e[0], e[1]]
        else:
            path_cmds += ["L", e[0], e[1]]

    path_cmds.append("z")
    path = dwg.path(path_cmds, stroke="black", fill="rgb(255,217,194)")
    dwg.add(path)
    dwg.save(pretty=True)


def svg_to_png(svg_path: Path, png_path: Path) -> bool:
    """
    Convert SVG to PNG using available libraries.
    Prefers CairoSVG (robust on macOS/Linux), falls back to svglib/reportlab.
    Returns True if conversion succeeded, False otherwise.
    """
    # Prefer CairoSVG when available (robust on macOS/Linux). Fallback to svglib/reportlab.
    if cairosvg is not None:
        try:
            cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
            return True
        except Exception:
            pass
    if svglib is not None and renderPM is not None:
        try:
            drawing = svglib.svg2rlg(str(svg_path))
            renderPM.drawToFile(drawing, str(png_path), fmt="PNG")
            return True
        except Exception:
            pass
    return False


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def choose_samples(
    items: List[Path],
    fraction: float,
    per_garment_max: int,
    per_garment_min: int,
    seed: Optional[int],
) -> List[Path]:
    if not items:
        return []
    n = len(items)
    k = int(max(per_garment_min, min(per_garment_max, math.ceil(fraction * n))))
    rng = random.Random(seed)
    return rng.sample(items, k=min(k, n))


def infer_garment_type(garment_folder_name: str) -> str:
    # Strip trailing _<digits> if present, otherwise return name
    parts = garment_folder_name.split("_")
    if parts and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return garment_folder_name


def process_dataset(
    data_root: Path,
    output_root: Path,
    sample_fraction: float,
    per_garment_max: int,
    per_garment_min: int,
    seed: Optional[int],
    output_format: str = "svg",
) -> None:
    if svgwrite is None:
        print(
            "Warning: svgwrite not found. Please install svgwrite to enable rendering."
        )
        return

    if output_format == "png":
        if svglib is None and renderPM is None and cairosvg is None:
            print(
                "Error: PNG format requested but no conversion library available. "
                "Please install cairosvg or svglib+reportlab."
            )
            return

    manifest_rows: List[List[str]] = []

    for garment_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        garment_type = infer_garment_type(garment_dir.name)

        # Datapoint folders (immediate children)
        datapoints = [p for p in garment_dir.iterdir() if p.is_dir()]
        if not datapoints:
            continue

        samples = choose_samples(
            datapoints, sample_fraction, per_garment_max, per_garment_min, seed
        )
        if not samples:
            continue

        print(
            f"Processing {len(samples)}/{len(datapoints)} samples from {garment_dir.name} (garment={garment_type})"
        )

        for dp in samples:
            spec_path = dp / "specification.json"
            if not spec_path.exists():
                continue
            try:
                spec = read_spec(spec_path, garment_type)
            except Exception as e:
                print(f"Failed to parse {spec_path}: {e}")
                continue

            for panel in spec.panels:
                # Compute integer cm dimensions
                width_cm = max(0, int(round(panel.width_cm)))
                height_cm = max(0, int(round(panel.height_cm)))

                # Output directory and filenames
                panel_dir = output_root / garment_type / panel.name
                safe_mkdir(panel_dir)

                base_name = f"panel_{width_cm}x{height_cm}_{spec.datapoint_id}"

                if output_format == "png":
                    # Render SVG to a temporary path, convert to PNG, then delete SVG
                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        prefix="panel_", suffix=".svg", delete=False
                    ) as tmp_svg_file:
                        tmp_svg_path = Path(tmp_svg_file.name)
                    try:
                        draw_panel_svg(panel, tmp_svg_path)
                    except Exception as e:
                        print(
                            f"Failed to render SVG for {spec.datapoint_id}:{panel.name}: {e}"
                        )
                        try:
                            tmp_svg_path.unlink(missing_ok=True)  # type: ignore
                        except Exception:
                            pass
                        continue

                    # Convert to PNG
                    png_out = panel_dir / f"{base_name}.png"
                    wrote_png = svg_to_png(tmp_svg_path, png_out)
                    try:
                        tmp_svg_path.unlink(missing_ok=True)  # type: ignore
                    except Exception:
                        pass
                    if not wrote_png:
                        print(
                            f"Skipping (PNG conversion unavailable or failed): {spec.datapoint_id} {panel.name}"
                        )
                        continue
                    out_path = str(png_out)

                else:  # svg format
                    svg_out = panel_dir / f"{base_name}.svg"
                    try:
                        draw_panel_svg(panel, svg_out)
                    except Exception as e:
                        print(
                            f"Failed to render SVG for {spec.datapoint_id}:{panel.name}: {e}"
                        )
                        continue
                    out_path = str(svg_out)

                manifest_rows.append(
                    [
                        spec.garment_type,
                        spec.datapoint_id,
                        panel.name,
                        str(width_cm),
                        str(height_cm),
                        str(spec_path),
                        out_path,
                    ]
                )

    # Write manifest CSV
    if manifest_rows:
        safe_mkdir(output_root)
        manifest_path = output_root / "panel_dimensions_manifest.csv"
        with manifest_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "garment_type",
                    "datapoint_id",
                    "panel",
                    "width_cm",
                    "height_cm",
                    "source_spec",
                    "output_path",
                ]
            )
            writer.writerows(manifest_rows)
        print(f"Saved manifest: {manifest_path}")
    else:
        print("No panels processed. Check your data path or sampling settings.")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-panel dimensions and export single-panel images"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("./data"),
        help="Root folder that contains garment-type dataset folders",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./images/shape"),
        help="Root output folder for pattern panel images and manifest",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.10,
        help="Fraction of datapoints to sample per garment-type (0..1)",
    )
    parser.add_argument(
        "--per-garment-max",
        type=int,
        default=100,
        help="Maximum datapoints per garment-type",
    )
    parser.add_argument(
        "--per-garment-min",
        type=int,
        default=10,
        help="Minimum datapoints per garment-type (if available)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducible sampling"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="svg",
        choices=["svg", "png"],
        help="Output image format: svg or png (default: svg)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()

    if not data_root.exists():
        print(f"Data root not found: {data_root}")
        sys.exit(1)

    process_dataset(
        data_root=data_root,
        output_root=output_root,
        sample_fraction=args.sample_fraction,
        per_garment_max=args.per_garment_max,
        per_garment_min=args.per_garment_min,
        seed=args.seed,
        output_format=args.format,
    )


if __name__ == "__main__":
    main()
