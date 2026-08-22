import json
from pathlib import Path

from modal_llm import experiment


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
