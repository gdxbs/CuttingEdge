# Original modules
from cutting_edge.cloth_recognition_module import ClothRecognitionModule
from cutting_edge.pattern_fitting_module import PatternFittingModule
from cutting_edge.pattern_recognition_module import PatternRecognitionModule

# Simplified modules
from cutting_edge.simple_cloth_recognition import ClothProcessor
from cutting_edge.simple_pattern_fitting import PatternFitter
from cutting_edge.simple_pattern_recognition import Pattern, PatternProcessor

__all__ = [
    # Original modules
    "ClothRecognitionModule",
    "PatternRecognitionModule",
    "PatternFittingModule",
    # Simplified modules
    "PatternProcessor",
    "Pattern",
    "ClothProcessor",
    "PatternFitter",
]
