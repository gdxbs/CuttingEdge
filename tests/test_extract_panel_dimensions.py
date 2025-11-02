"""
Tests for extract_panel_dimensions.py

This test verifies that the script can generate both SVG and PNG output formats
from panel specification files.
"""

import json
import tempfile
from pathlib import Path

import pytest


def create_test_specification(test_dir: Path, datapoint_name: str) -> Path:
    """Create a test specification file with a simple rectangular panel."""
    dp_dir = test_dir / datapoint_name
    dp_dir.mkdir(parents=True, exist_ok=True)

    spec = {
        "pattern": {
            "panels": {
                "front": {
                    "vertices": [[0.0, 0.0], [50.0, 0.0], [50.0, 80.0], [0.0, 80.0]],
                    "edges": [
                        {"endpoints": [0, 1]},
                        {"endpoints": [1, 2]},
                        {"endpoints": [2, 3]},
                        {"endpoints": [3, 0]},
                    ],
                }
            }
        }
    }

    spec_path = dp_dir / "specification.json"
    with spec_path.open("w") as f:
        json.dump(spec, f)

    return dp_dir


def test_svg_format_output():
    """Test that SVG format produces .svg files."""
    import extract_panel_dimensions as epd

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create test data
        data_root = tmp_path / "data" / "shirt_001"
        create_test_specification(data_root, "test_dp_001")

        # Create output directory
        output_root = tmp_path / "output_svg"

        # Process with SVG format
        epd.process_dataset(
            data_root=data_root.parent,
            output_root=output_root,
            sample_fraction=1.0,
            per_garment_max=10,
            per_garment_min=1,
            seed=42,
            output_format="svg",
        )

        # Verify SVG file was created
        svg_files = list(output_root.glob("**/*.svg"))
        assert len(svg_files) > 0, "No SVG files were created"
        assert svg_files[0].name == "panel_50x80_test_dp_001.svg"


def test_png_format_output():
    """Test that PNG format produces .png files (requires conversion library)."""
    import extract_panel_dimensions as epd

    # Skip if no PNG conversion libraries available
    if epd.cairosvg is None and (epd.svglib is None or epd.renderPM is None):
        pytest.skip("PNG conversion libraries not available")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create test data
        data_root = tmp_path / "data" / "shirt_001"
        create_test_specification(data_root, "test_dp_001")

        # Create output directory
        output_root = tmp_path / "output_png"

        # Process with PNG format
        epd.process_dataset(
            data_root=data_root.parent,
            output_root=output_root,
            sample_fraction=1.0,
            per_garment_max=10,
            per_garment_min=1,
            seed=42,
            output_format="png",
        )

        # Verify PNG file was created
        png_files = list(output_root.glob("**/*.png"))
        assert len(png_files) > 0, "No PNG files were created"
        assert png_files[0].name == "panel_50x80_test_dp_001.png"


def test_manifest_created():
    """Test that the manifest CSV is created."""
    import extract_panel_dimensions as epd

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create test data
        data_root = tmp_path / "data" / "shirt_001"
        create_test_specification(data_root, "test_dp_001")

        # Create output directory
        output_root = tmp_path / "output"

        # Process dataset
        epd.process_dataset(
            data_root=data_root.parent,
            output_root=output_root,
            sample_fraction=1.0,
            per_garment_max=10,
            per_garment_min=1,
            seed=42,
            output_format="svg",
        )

        # Verify manifest was created
        manifest_path = output_root / "panel_dimensions_manifest.csv"
        assert manifest_path.exists(), "Manifest CSV was not created"

        # Read and verify manifest content
        with manifest_path.open("r") as f:
            lines = f.readlines()
            assert len(lines) == 2, "Manifest should have header + 1 data row"
            assert "garment_type,datapoint_id,panel" in lines[0]
            assert "shirt,test_dp_001,front,50,80" in lines[1]
