"""
Statistics module for CRBench.
"""

from crbench.statistics.bootstrap import BootstrapEngine, BootstrapCIResult
from crbench.statistics.hypothesis import HypothesisEngine, PairedComparisonResult
from crbench.statistics.stability import StabilityEngine, RankingStabilityResult
from crbench.statistics.sensitivity import WeightingSensitivityAnalyzer, WeightingSensitivityResult

# Aliases for convenience
BootstrapResult = BootstrapCIResult
PermutationTestResult = PairedComparisonResult
StabilityResult = RankingStabilityResult

__all__ = [
    "BootstrapEngine",
    "BootstrapCIResult",
    "BootstrapResult",
    "HypothesisEngine",
    "PairedComparisonResult",
    "PermutationTestResult",
    "StabilityEngine",
    "RankingStabilityResult",
    "StabilityResult",
    "WeightingSensitivityAnalyzer",
    "WeightingSensitivityResult",
]
