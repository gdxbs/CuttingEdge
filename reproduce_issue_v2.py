
import logging
import sys
import os
import numpy as np
from cutting_edge.pattern_recognition_module import PatternRecognitionModule
from cutting_edge.cloth_recognition_module import ClothRecognitionModule
from cutting_edge.pattern_fitting_module import PatternFittingModule
from cutting_edge.config import FITTING

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_test():
    logger.info("Starting reproduction test v2 (Real Data)...")
    
    # Initialize modules
    pt_module = PatternRecognitionModule()
    cl_module = ClothRecognitionModule()
    fitter = PatternFittingModule()
    
    # Paths
    # Note: Using absolute paths based on previous `ls` output
    cloth_path = r"D:\cut\CuttingEdge\images\cloth\freeform\cloth_23_510x553.jpg"
    pattern_path = r"D:\cut\CuttingEdge\images\shape\dress_sleeveless\skirt_back\panel_101x72_dress_sleeveless_5BZP3PVJSQ.png"
    
    if not os.path.exists(cloth_path):
        logger.error(f"Cloth file not found: {cloth_path}")
        return
    if not os.path.exists(pattern_path):
         logger.error(f"Pattern file not found: {pattern_path}")
         return
         
    # Load data
    logger.info("Loading cloth and pattern...")
    cloth = cl_module.process_image(cloth_path)
    pattern = pt_module.process_image(pattern_path)
    
    logger.info(f"Cloth: {cloth.width}x{cloth.height}")
    logger.info(f"Pattern: {pattern.width}x{pattern.height}")

    # Force strict baseline mode for fairness
    search_res = 5.0
    
    logger.info("--- Test 1: Baseline placement with ORIGINAL pattern ---")
    placement1 = fitter._find_baseline_placement(pattern, cloth, [], search_resolution=search_res)
    if placement1:
        logger.info(f"SUCCESS: Original pattern placed at {placement1.position}")
    else:
        logger.error("FAILURE: Original pattern NOT placed")

    logger.info("--- Test 2: Baseline placement with SCALED (x1.0) pattern ---")
    scaled_pattern = fitter.scale_pattern(pattern, 1.0)
    placement2 = fitter._find_baseline_placement(scaled_pattern, cloth, [], search_resolution=search_res)
    if placement2:
        logger.info(f"SUCCESS: Scaled pattern placed at {placement2.position}")
    else:
        logger.error("FAILURE: Scaled pattern NOT placed")
        
    if placement1 and not placement2:
        logger.error("ISSUE REPRODUCED: Original fits but Scaled(x1.0) fails!")
    elif not placement1 and placement2:
        logger.warning("ISSUE REPRODUCED (INVERSE): Original fails but Scaled(x1.0) fits!")
        # This confirms that scale_pattern(p, 1.0) produces a 'better' pattern than the original one
    elif placement1 and placement2:
         logger.info("Both fit. Issue NOT reproduced with this data.")
    else:
         logger.info("Neither fit. Issue NOT reproduced.")

if __name__ == "__main__":
    run_test()
