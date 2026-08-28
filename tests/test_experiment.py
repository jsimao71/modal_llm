import json
from pathlib import Path
from types import SimpleNamespace

import torch

from modal_llm import experiment
from modal_llm.data import ConstraintDataset
from modal_llm.model import ModeTransformer


def test_auto_device_prefers_cuda_then_xpu_then_cpu(monkeypatch) -> None:
    fake_xpu = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setattr(torch, "xpu", fake_xpu, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert experiment._device("auto").type == "xpu"
    assert experiment._device("xpu").type == "xpu"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert experiment._device("auto").type == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    fake_xpu.is_available = lambda: False
    assert experiment._device("auto").type == "cpu"


def test_explicit_xpu_fails_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        torch, "xpu", SimpleNamespace(is_available=lambda: False), raising=False
    )
    try:
        experiment._device("xpu")
    except RuntimeError as error:
        assert str(error) == "XPU requested but unavailable"
    else:
        raise AssertionError("Expected unavailable explicit XPU to fail")


def test_task_exact_requires_active_facets_and_end_token() -> None:
    target = torch.tensor([[20, 21, 0, 4], [20, 21, 0, 4]])
    active = torch.tensor([[True, True, False], [True, True, False]])
    generated = torch.tensor([[20, 21, 99, 4], [20, 21, 0, 3]])

    exact = experiment._task_exact(generated, target, active, max_facets=3, out_end=4)

    assert exact.tolist() == [True, False]


def test_horizon_evaluation_reports_quartiles_and_drift() -> None:
    dataset = ConstraintDataset(
        4,
        seed=23,
        split="test",
        min_facets=6,
        max_facets=6,
        goal_prompt_style="canonical",
        target_repetitions=2,
    )
    model = ModeTransformer(
        dataset.vocab,
        "B5",
        d_model=16,
        nhead=4,
        layers=1,
        dropout=0.0,
        max_length=64,
        generation_prompt_only=True,
    )
    config = {"evaluation": {"batch_size": 4}}

    metrics, predictions = experiment.evaluate_horizon(
        model, dataset, config, torch.device("cpu")
    )

    assert len(predictions) == 4
    assert metrics["target_output_length"] == 13.0
    assert "task_quartile_4_satisfaction" in metrics
    assert "shuffled_goal_drift" in metrics

    aligned = ModeTransformer(
        dataset.vocab,
        "B5",
        d_model=16,
        nhead=4,
        layers=1,
        dropout=0.0,
        max_length=64,
        generation_prompt_only=True,
        goal_vectors=6,
        goal_pooling="facet_tokens",
        conditioning_mode="slot_prefix",
        prefix_tokens=6,
    )
    aligned_metrics, aligned_predictions = experiment.evaluate_horizon(
        aligned, dataset, config, torch.device("cpu")
    )
    assert len(aligned_predictions) == 4
    assert "facet_slot_swap_exact_match" in aligned_metrics
    assert "facet_slot_swap_selected_satisfaction" in aligned_metrics
    assert "facet_slot_swap_untouched_satisfaction" in aligned_metrics

    direct = ModeTransformer(
        dataset.vocab,
        "B0",
        d_model=16,
        nhead=4,
        layers=1,
        dropout=0.0,
        max_length=64,
    )
    direct_metrics, direct_predictions = experiment.evaluate_horizon(
        direct, dataset, config, torch.device("cpu")
    )
    assert len(direct_predictions) == 4
    assert "goal_drift" in direct_metrics
    assert "shuffled_goal_drift" not in direct_metrics


def test_suite_summary_omits_all_nonfinite_metrics(tmp_path: Path, monkeypatch) -> None:
    def fake_run(config, output_root, repository):
        run_dir = (
            output_root
            / config["experiment"]
            / config["baseline"]
            / f"run-seed{config['seed']}"
        )
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "baseline": config["baseline"],
                    "seed": config["seed"],
                    "finite_metric": 1.0,
                    "all_nonfinite_metric": float("nan"),
                }
            ),
            encoding="utf-8",
        )
        return run_dir

    monkeypatch.setattr(experiment, "run", fake_run)
    experiment.run_suite(
        {"experiment": "suite-test", "baselines": ["B3", "B2"], "seeds": [0]},
        tmp_path,
        tmp_path,
    )

    summaries = list((tmp_path / "suite-test").glob("suite-*.json"))
    assert len(summaries) == 1
    payload = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert "all_nonfinite_metric" not in payload["groups"]["B3"]
    assert payload["groups"]["B3"]["finite_metric"]["mean"] == 1.0
    assert (
        payload["paired_comparisons"]["B3-minus-B2"]["finite_metric"]
        ["mean_difference"]
        == 0.0
    )
