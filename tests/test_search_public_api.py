from hyperphoenixcv.search_grid import ExhaustiveSearchStrategy as GridImplementation
from hyperphoenixcv.search_optuna import OptunaSearchStrategy as OptunaImplementation
from hyperphoenixcv.search_random import RandomSearchStrategy as RandomImplementation
from hyperphoenixcv.search_strategies import (
    ExhaustiveSearchStrategy,
    OptunaSearchStrategy,
    RandomSearchStrategy,
)
from hyperphoenixcv.search_surrogate import ExperimentalSurrogateRankingStrategy as SurrogateImplementation
from hyperphoenixcv.search_strategies import ExperimentalSurrogateRankingStrategy


def test_legacy_search_strategy_imports_reexport_concrete_implementations():
    assert ExhaustiveSearchStrategy is GridImplementation
    assert RandomSearchStrategy is RandomImplementation
    assert OptunaSearchStrategy is OptunaImplementation
    assert ExperimentalSurrogateRankingStrategy is SurrogateImplementation
