# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import ParsingPartWBCPredictor
from .train import ParsingPartWBCTrainer
from .val import ParsingPartWBCValidator

__all__ = "ParsingPartWBCPredictor", "ParsingPartWBCTrainer", "ParsingPartWBCValidator"
