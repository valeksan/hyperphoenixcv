"""Release-gate checks for published Markdown and its runnable core workflow."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from hyperphoenixcv import HyperPhoenixCV
from hyperphoenixcv.audit import TrialHistory


ROOT = Path(__file__).resolve().parents[1]


def _python_blocks(path: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.DOTALL)


class _FastReadmeExamples(ast.NodeTransformer):
    """Keep published examples intact while making their test execution small."""

    @staticmethod
    def _small_space(value: ast.expr) -> ast.expr:
        if not isinstance(value, ast.Dict):
            return value
        values: list[ast.expr] = []
        for item in value.values:
            if isinstance(item, ast.List) and item.elts:
                values.append(ast.List(elts=[item.elts[0]], ctx=ast.Load()))
            else:
                values.append(item)
        return ast.Dict(keys=value.keys, values=values)

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        self.generic_visit(node)
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"search_space", "param_grid"}
        ):
            node.value = self._small_space(node.value)
        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "make_classification":
            node.keywords = [
                keyword if keyword.arg != "n_samples" else ast.keyword(
                    arg="n_samples", value=ast.Constant(40)
                )
                for keyword in node.keywords
            ]
        if isinstance(node.func, ast.Name) and node.func.id in {"TimeSeriesSplit", "GroupKFold"}:
            node.keywords = [
                keyword if keyword.arg != "n_splits" else ast.keyword(
                    arg="n_splits", value=ast.Constant(2)
                )
                for keyword in node.keywords
            ]
        if isinstance(node.func, ast.Name) and node.func.id == "HyperPhoenixCV":
            small_keywords: list[ast.keyword] = []
            for keyword in node.keywords:
                if keyword.arg in {"cv", "n_trials", "n_jobs"}:
                    small_keywords.append(ast.keyword(arg=keyword.arg, value=ast.Constant(1 if keyword.arg != "cv" else 2)))
                elif keyword.arg == "search_space":
                    small_keywords.append(ast.keyword(arg=keyword.arg, value=self._small_space(keyword.value)))
                else:
                    small_keywords.append(keyword)
            node.keywords = small_keywords
        return node


def _exec_readme_block(block: str, namespace: dict[str, object]) -> None:
    tree = _FastReadmeExamples().visit(ast.parse(block))
    ast.fix_missing_locations(tree)
    exec(compile(tree, "README example", "exec"), namespace)


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


def _execute_readme_examples(tmp_path: Path, monkeypatch, *, include_optuna: bool) -> None:
    """Execute README blocks with reduced test-only workloads."""
    for readme_name in ("README.md", "README_RU.md"):
        workdir = tmp_path / readme_name
        workdir.mkdir()
        monkeypatch.chdir(workdir)
        namespace: dict[str, object] = {"__name__": "__readme_example__"}
        for block in _python_blocks(ROOT / readme_name):
            if not include_optuna and "optuna" in block.lower():
                continue
            if "history = hp.trial_history_" in block:
                namespace["hp"] = namespace["fitted_hp"]
            if "param_grid" in block and "param_grid" not in namespace:
                namespace["param_grid"] = namespace["search_space"]
            if "GroupKFold" in block:
                import numpy as np
                from sklearn.ensemble import RandomForestClassifier

                namespace["groups"] = np.arange(len(namespace["y"])) // 2
                namespace["model"] = RandomForestClassifier(random_state=42)
            if "refit=False" in block and "optuna_directions" in block:
                import optuna
                from sklearn.linear_model import LogisticRegression

                namespace["model"] = LogisticRegression(max_iter=100)
                namespace["space"] = {
                    "C": optuna.distributions.FloatDistribution(0.1, 1.0)
                }
            if "export_parquet" in block:
                with pytest.raises(ImportError, match=r"hyperphoenixcv\[parquet\]"):
                    _exec_readme_block(block, namespace)
            else:
                _exec_readme_block(block, namespace)
            if "Best parameters" in block or "Лучшие параметры" in block:
                namespace["fitted_hp"] = namespace["hp"]


def test_core_readme_python_examples_execute(tmp_path: Path, monkeypatch) -> None:
    """Execute every non-Optuna README Python block in core install."""
    _execute_readme_examples(tmp_path, monkeypatch, include_optuna=False)


def test_optuna_readme_python_examples_execute(tmp_path: Path, monkeypatch) -> None:
    """Execute every README Python block when optional Optuna is installed."""
    pytest.importorskip("optuna")
    _execute_readme_examples(tmp_path, monkeypatch, include_optuna=True)


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
