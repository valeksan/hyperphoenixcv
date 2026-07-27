"""Release-gate checks for published Markdown and its runnable core workflow."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from hyperphoenixcv import HyperPhoenixCV
from hyperphoenixcv.audit import TrialHistory


ROOT = Path(__file__).resolve().parents[1]


def _python_blocks(path: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.DOTALL)


def test_readme_python_examples_parse() -> None:
    """Every published Python README snippet remains executable Python syntax."""
    for readme in (ROOT / "README.md", ROOT / "README_RU.md"):
        blocks = _python_blocks(readme)
        assert blocks, readme
        for number, block in enumerate(blocks, start=1):
            ast.parse(block, filename=f"{readme.name}:block-{number}")


def test_readme_quick_start_and_resume(tmp_path: Path) -> None:
    """Execute documented core grid/resume workflow against temporary storage."""
    X, y = make_classification(
        n_samples=40, n_features=6, n_informative=3, random_state=42,
    )
    kwargs = dict(
        estimator=LogisticRegression(max_iter=200),
        search_space={"C": [0.1]},
        strategy="grid",
        scoring="accuracy",
        cv=2,
        storage_path=tmp_path / "my_experiment.sqlite3",
        results_csv=tmp_path / "my_experiment_results.csv",
        dataset_id="training-data-v1",
        verbose=False,
    )
    first = HyperPhoenixCV(**kwargs).fit(X, y)
    resumed = HyperPhoenixCV(**kwargs, resume="must").fit(X, y)
    assert first.best_params_ == resumed.best_params_
    assert len(resumed.cv_results_["params"]) == 1


def test_api_reference_tracks_public_signatures() -> None:
    reference = (ROOT / "docs" / "api_reference.md").read_text(encoding="utf-8")
    for parameter in inspect.signature(HyperPhoenixCV).parameters:
        assert parameter in reference
    for method in (
        "fit", "get_top_results", "load_results_from_checkpoint",
        "load_trial_history", "clear_storage", "import_legacy_checkpoint",
    ):
        assert f"`{method}" in reference
    for method in ("count", "page", "iter_records", "export_json", "export_csv", "export_parquet"):
        assert f"`{method}" in reference


def test_guide_links_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in (
        "resume_and_storage.md", "refit_objectives.md", "pruning.md",
        "parallelism.md", "audit_and_events.md", "api_reference.md",
    ):
        assert name in readme
        assert (ROOT / "docs" / name).is_file()
