
import logging
import sys
import os
import numpy as np
from cutting_edge.pattern_recognition_module import Pattern
from cutting_edge.cloth_recognition_module import ClothMaterial
from cutting_edge.pattern_fitting_module import PatternFittingModule
from cutting_edge.config import FITTING

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_test():
    logger.info("Starting reproduction test...")
    
    # Initialize fitting module
    fitter = PatternFittingModule()
    
    # Create a dummy irregular cloth (T-shape)
    # 0,0 -> 100,0 -> 100,100 -> 0,100 is bounding box
    # Cut out top corners to make it T-shaped or similar
    # Let's make a U-shape: 100x100 but with a block taken out of top middle
    # 0,0 -> 100,0 -> 100,100 -> 70,100 -> 70,30 -> 30,30 -> 30,100 -> 0,100
    contour = np.array([
        [0, 0], [100, 0], [100, 100], [70, 100], 
        [70, 30], [30, 30], [30, 100], [0, 100]
    ], dtype=float)
    
    cloth = ClothMaterial(
        id=1, name="test_cloth", cloth_type="cotton",
        width=100.0, height=100.0,
        total_area=10000.0 - (40*70), # 100x100 - 40x70 cutout
        usable_area=7200.0,
        contour=contour,
        defects=[]
    )
    
    # Create a pattern that fits in the "U" part (bottom area)
    # Width 80, Height 20. Should fit at bottom (y=0 to 20 approx)
    pattern = Pattern(
        id=1, name="long_strip", pattern_type="other",
        width=80.0, height=20.0,
        area=1600.0,
        contour=np.array([[0,0], [80,0], [80,20], [0,20]], dtype=float),
        confidence=1.0
    )
    
    logger.info("--- Test 1: Baseline placement with ORIGINAL pattern ---")
    placement1 = fitter._find_baseline_placement(pattern, cloth, [], search_resolution=5.0)
    if placement1:
        logger.info(f"SUCCESS: Original pattern placed at {placement1.position}")
    else:
        logger.error("FAILURE: Original pattern NOT placed")

    logger.info("--- Test 2: Baseline placement with SCALED (x1.0) pattern ---")
    scaled_pattern = fitter.scale_pattern(pattern, 1.0)
    placement2 = fitter._find_baseline_placement(scaled_pattern, cloth, [], search_resolution=5.0)
    if placement2:
        logger.info(f"SUCCESS: Scaled pattern placed at {placement2.position}")
    else:
        logger.error("FAILURE: Scaled pattern NOT placed")
        
    # Compare attributes
    logger.info("--- Comparing Pattern Attributes ---")
    logger.info(f"Original ID: {pattern.id}, Scaled ID: {scaled_pattern.id}")
    logger.info(f"Original Name: {pattern.name}, Scaled Name: {scaled_pattern.name}")
    logger.info(f"Original Contour shape: {pattern.contour.shape if pattern.contour is not None else 'None'}")
    logger.info(f"Scaled Contour shape: {scaled_pattern.contour.shape if scaled_pattern.contour is not None else 'None'}")
    
    if placement1 and not placement2:
        logger.error("ISSUE REPRODUCED: Original fits but Scaled(x1.0) fails!")
    elif not placement1 and placement2:
        logger.error("ISSUE REPRODUCED: Original fails but Scaled(x1.0) fits (Inverse problem)!")
    elif placement1 and placement2:
         logger.info("Both fit. Issue NOT reproduced with this simple case.")
    else:
         logger.info("Neither fit. Issue NOT reproduced.")

if __name__ == "__main__":
    run_test()
