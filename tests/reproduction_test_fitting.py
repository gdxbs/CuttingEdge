
import logging
import numpy as np
import sys
import os

# Ensure we can import from local modules
sys.path.append(os.getcwd())

from cutting_edge.pattern_recognition_module import Pattern
from cutting_edge.cloth_recognition_module import ClothMaterial
from cutting_edge.pattern_fitting_module import PatternFittingModule
from cutting_edge.config import FITTING

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fallback_logic():
    # 1. Create a dummy cloth (100x100)
    cloth = ClothMaterial(
        id=1,
        name="test_cloth",
        cloth_type="cotton",
        width=100.0,
        height=100.0,
        total_area=10000.0,
        usable_area=10000.0,
        contour=np.array([[[0, 0]], [[100, 0]], [[100, 100]], [[0, 100]]], dtype=float),
        defects=[]
    )

    # 2. Create a dummy pattern (50x50)
    pattern = Pattern(
        id=1,
        name="test_pattern",
        pattern_type="other",
        width=50.0,
        height=50.0,
        area=2500.0,
        contour=np.array([[[0, 0]], [[50, 0]], [[50, 50]], [[0, 50]]], dtype=float),
        confidence=1.0
    )

    # 3. Initialize Fitting Module
    # Need to mock model_path or ensure it doesn't crash if missing
    # The init tries to load a model if path is provided or defaults. 
    # If we pass None, it defaults. If file missing, it just logs warning (based on code reading).
    fitter = PatternFittingModule()

    # 4. Monkey patch FITTING config to force sparse grid search failure
    # We want find_best_placement to fail its initial search.
    original_sample_size = FITTING["GRID_SAMPLE_SIZE"]
    # Set to 0 so no positions are tried in the main loop
    FITTING["GRID_SAMPLE_SIZE"] = 0 
    
    logger.info("Starting test with GRID_SAMPLE_SIZE=0 to force initial search failure...")
    
    try:
        # 5. Fit pattern
        # result = fitter.fit_patterns([pattern], cloth, baseline_mode=False)
        # Calling fit_patterns which calls find_best_placement
        result = fitter.fit_patterns([pattern], cloth, baseline_mode=False)
        
        # 6. Check results
        if result["patterns_placed"] == 1:
            logger.info("SUCCESS: Pattern placed via fallback mechanism!")
            # Verify it was placed
            placement = result["placements"][0]
            logger.info(f"Placement Score: {placement.score}")
            if placement.score > 0 and placement.score != 1.0: # 1.0 is the dummy score, we want calculated score
                 logger.info("Score verification: Score was calculated correctly (not dummy global 1.0 from baseline)")
            elif placement.score == 1.0:
                 logger.warning("Score verification: Score is 1.0, might be dummy score not updated (or coincidence)")
                 
        else:
            logger.error("FAILURE: Pattern was not placed.")
            sys.exit(1)
            
    finally:
        # Restore config
        FITTING["GRID_SAMPLE_SIZE"] = original_sample_size

if __name__ == "__main__":
    test_fallback_logic()
