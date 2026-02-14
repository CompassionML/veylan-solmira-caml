"""
Persona Vectors: Activation Steering for Trait Expression

A package for extracting and applying persona vectors to steer
language model behavior toward specific traits like compassion,
open-mindedness, etc.

Based on the Contrastive Activation Addition (CAA) methodology.
"""

from .config import Config
from .artifacts import TRAIT_DEFINITIONS, COMPASSION_ARTIFACTS, OPEN_MINDEDNESS_ARTIFACTS, NON_HELPFULNESS_ARTIFACTS
from .model import ModelManager
from .steering import SteeringManager
from .extraction import VectorExtractor
from .evaluation import (
    LayerSelector,
    GeminiJudge,
    ScaleSweepEvaluator,
    VectorEffectivenessTester,
    BaselineBehaviorChecker,
)
from .controls import ControlExperiments
from .stats import StatisticalValidator, compute_statistics, print_statistics_report

__version__ = "0.1.0"
__all__ = [
    "Config",
    "TRAIT_DEFINITIONS",
    "COMPASSION_ARTIFACTS",
    "OPEN_MINDEDNESS_ARTIFACTS",
    "NON_HELPFULNESS_ARTIFACTS",
    "ModelManager",
    "SteeringManager",
    "VectorExtractor",
    "LayerSelector",
    "GeminiJudge",
    "ScaleSweepEvaluator",
    "VectorEffectivenessTester",
    "BaselineBehaviorChecker",
    "ControlExperiments",
    "StatisticalValidator",
    "compute_statistics",
    "print_statistics_report",
]
