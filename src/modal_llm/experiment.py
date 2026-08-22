"""Training, evaluation, and baseline-suite entrypoint for Paper 1."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from .common import atomic_write_json, capture_provenance, seed_everything
from .common.reproducibility import stable_hash
from .data import ConstraintDataset, collate_examples
from .metrics import average_precision, binary_auroc, brier_score, expected_calibration_error
from .model import BASELINES, ModeTransformer


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _dataset(config: dict[str, Any], split: str, seed: int) -> ConstraintDataset:
    data = config["data"]
    size_key = {"train": "train_size", "validation": "validation_size", "test": "test_size"}[split]
    return ConstraintDataset(
        int(data[size_key]), seed=seed, split=split,
        max_facets=int(data.get("max_facets", 4)),
        min_facets=int(data.get("min_facets", 2)),
        max_distractors=int(data.get("max_distractors", 2)),
        max_filler=int(data.get("max_filler", 3)),
    )


def _model(config: dict[str, Any], dataset: ConstraintDataset) -> ModeTransformer:
    values = config["model"]
    return ModeTransformer(
        dataset.vocab, str(config["baseline"]),
        d_model=int(values.get("d_model", 64)), nhead=int(values.get("nhead", 4)),
        layers=int(values.get("layers", 2)), ff_mult=int(values.get("ff_mult", 4)),
        dropout=float(values.get("dropout", 0.1)),
        max_length=int(values.get("max_length", 128)),
    )


def train(
    model: ModeTransformer,
    dataset: ConstraintDataset,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, float]], int]:
    values = config["training"]
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader = DataLoader(
        dataset, batch_size=int(values.get("batch_size", 32)), shuffle=True,
        generator=generator, collate_fn=collate_examples,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(values.get("learning_rate", 3e-4)),
        weight_decay=float(values.get("weight_decay", 0.01)),
    )
    weights = dict(values.get("loss_weights") or {})
    history: list[dict[str, float]] = []
    updates = 0
    model.reset_compute_stats()
    model.train()
    for epoch in range(int(values.get("epochs", 5))):
        totals: dict[str, list[float]] = defaultdict(list)
        started = time.perf_counter()
        for raw in loader:
            batch = _move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            losses = model.losses(batch, weights)
            losses["total"].backward()
            clip_grad_norm_(model.parameters(), float(values.get("grad_clip", 1.0)))
            optimizer.step()
            updates += 1
            for key, value in losses.items():
                totals[key].append(float(value.detach()))
        row = {f"train_{key}": mean(items) for key, items in totals.items()}
        row.update(epoch=epoch + 1, updates=updates, wall_seconds=time.perf_counter() - started)
        history.append(row)
        print(json.dumps(row, sort_keys=True))
    return history, updates


@torch.no_grad()
def validation_losses(
    model: ModeTransformer,
    dataset: ConstraintDataset,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    loader = DataLoader(
        dataset, batch_size=int(config["evaluation"].get("batch_size", 64)),
        shuffle=False, collate_fn=collate_examples,
    )
    weights = dict(config["training"].get("loss_weights") or {})
    totals: dict[str, list[float]] = defaultdict(list)
    model.eval()
    for raw in loader:
        losses = model.losses(_move(raw, device), weights)
        for key, value in losses.items():
            totals[key].append(float(value))
    return {f"validation_{key}": mean(values) for key, values in totals.items()}


@torch.no_grad()
def evaluate(
    model: ModeTransformer,
    dataset: ConstraintDataset,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    loader = DataLoader(
        dataset, batch_size=int(config["evaluation"].get("batch_size", 64)),
        shuffle=False, collate_fn=collate_examples,
    )
    model.eval()
    exact: list[float] = []
    facets: list[float] = []
    labels: list[int] = []
    scores: list[float] = []
    rankings: list[float] = []
    zero_exact: list[float] = []
    shuffle_exact: list[float] = []
    paraphrase_goal_exact: list[float] = []
    facet_substitution_success: list[float] = []
    no_mode_exact: list[float] = []
    swapped_mode_exact: list[float] = []
    paraphrase_similarity: list[float] = []
    different_goal_similarity: list[float] = []
    predictions: list[dict[str, Any]] = []
    base_calls = base_positions = 0
    intervention_calls = intervention_positions = 0
    validation_calls = validation_positions = 0
    model.reset_compute_stats()
    started = time.perf_counter()
    for raw in loader:
        batch = _move(raw, device)
        before_base = model.compute_stats()
        generated, goal, _ = model.generate(batch["prompt"], batch["target"].shape[1])
        after_base = model.compute_stats()
        base_calls += after_base["forward_calls"] - before_base["forward_calls"]
        base_positions += after_base["token_positions"] - before_base["token_positions"]
        active = batch["active_facets"].bool()
        token_ok = generated[:, : model.vocab.max_facets].eq(
            batch["target"][:, : model.vocab.max_facets]
        )
        facet_fraction = (token_ok & active).sum(1).float() / active.sum(1)
        batch_exact = ((token_ok | ~active).all(1) & generated[:, -1].eq(model.vocab.OUT_END))
        exact.extend(batch_exact.float().cpu().tolist())
        facets.extend(facet_fraction.cpu().tolist())

        if model.spec.use_goal and goal is not None:
            zeros = torch.zeros_like(goal)
            zero_generated, _, zero_calls = model.generate(
                batch["prompt"], batch["target"].shape[1], forced_goal=zeros
            )
            shuffled = goal.roll(1, 0)
            shuffled_generated, _, shuffle_calls = model.generate(
                batch["prompt"], batch["target"].shape[1], forced_goal=shuffled
            )
            zero_ok = zero_generated[:, : model.vocab.max_facets].eq(
                batch["target"][:, : model.vocab.max_facets]
            )
            shuffle_ok = shuffled_generated[:, : model.vocab.max_facets].eq(
                batch["target"][:, : model.vocab.max_facets]
            )
            zero_exact.extend((zero_ok | ~active).all(1).float().cpu().tolist())
            shuffle_exact.extend((shuffle_ok | ~active).all(1).float().cpu().tolist())
            indices = [int(example_id.rsplit("-", 1)[1]) for example_id in raw["ids"]]
            paraphrases = pad_sequence(
                [dataset.paraphrase(index) for index in indices], batch_first=True,
                padding_value=dataset.vocab.PAD,
            ).to(device)
            _, paraphrase_goal = model.encode(paraphrases)
            paraphrase_similarity.extend(
                torch.nn.functional.cosine_similarity(goal, paraphrase_goal).cpu().tolist()
            )
            if len(goal) > 1:
                different_goal_similarity.extend(
                    torch.nn.functional.cosine_similarity(goal, goal.roll(1, 0)).cpu().tolist()
                )
            paraphrase_generated, _, _ = model.generate(
                batch["prompt"], batch["target"].shape[1], forced_goal=paraphrase_goal
            )
            paraphrase_ok = paraphrase_generated[:, : model.vocab.max_facets].eq(
                batch["target"][:, : model.vocab.max_facets]
            )
            paraphrase_goal_exact.extend((paraphrase_ok | ~active).all(1).float().cpu().tolist())

            _, counterfactual_goal = model.encode(batch["counterfactual_prompt"])
            substituted, _, _ = model.generate(
                batch["prompt"], batch["target"].shape[1], forced_goal=counterfactual_goal
            )
            row_index = torch.arange(len(substituted), device=device)
            changed = substituted[
                row_index, batch["corrupt_facet"]
            ].eq(batch["counterfactual_target"][row_index, batch["corrupt_facet"]])
            facet_substitution_success.extend(changed.float().cpu().tolist())

            mode_weights = model.mode_embedding.weight.detach().clone()
            model.mode_embedding.weight.zero_()
            without_mode, _, _ = model.generate(batch["prompt"], batch["target"].shape[1])
            model.mode_embedding.weight.copy_(mode_weights)
            no_mode_ok = without_mode[:, : model.vocab.max_facets].eq(
                batch["target"][:, : model.vocab.max_facets]
            )
            no_mode_exact.extend((no_mode_ok | ~active).all(1).float().cpu().tolist())

            swapped = mode_weights.clone()
            swapped[[0, 1]] = swapped[[1, 0]]
            model.mode_embedding.weight.copy_(swapped)
            swapped_mode, _, _ = model.generate(batch["prompt"], batch["target"].shape[1])
            model.mode_embedding.weight.copy_(mode_weights)
            swapped_ok = swapped_mode[:, : model.vocab.max_facets].eq(
                batch["target"][:, : model.vocab.max_facets]
            )
            swapped_mode_exact.extend((swapped_ok | ~active).all(1).float().cpu().tolist())

        after_interventions = model.compute_stats()
        intervention_calls += after_interventions["forward_calls"] - after_base["forward_calls"]
        intervention_positions += (
            after_interventions["token_positions"] - after_base["token_positions"]
        )

        positive_probability = negative_probability = None
        if model.spec.validator is not None:
            positive, _ = model.validation_logits(batch["prompt"], batch["target"], goal)
            negative, _ = model.validation_logits(batch["prompt"], batch["corrupted"], goal)
            positive_probability = positive.sigmoid()
            negative_probability = negative.sigmoid()
            labels.extend([1] * len(positive) + [0] * len(negative))
            scores.extend(positive_probability.cpu().tolist() + negative_probability.cpu().tolist())
            rankings.extend((positive > negative).float().cpu().tolist())

        after_validation = model.compute_stats()
        validation_calls += after_validation["forward_calls"] - after_interventions["forward_calls"]
        validation_positions += (
            after_validation["token_positions"] - after_interventions["token_positions"]
        )

        for index, example_id in enumerate(raw["ids"]):
            row = {
                "id": example_id,
                "target": raw["target"][index].tolist(),
                "generated": generated[index].cpu().tolist(),
                "facet_fraction": float(facet_fraction[index]),
                "exact": bool(batch_exact[index]),
            }
            if positive_probability is not None and negative_probability is not None:
                row.update(
                    positive_score=float(positive_probability[index]),
                    near_miss_score=float(negative_probability[index]),
                )
            predictions.append(row)

    elapsed = time.perf_counter() - started
    compute = model.compute_stats()
    metrics = {
        "task_exact_match": mean(exact),
        "task_facet_satisfaction": mean(facets),
        "evaluation_examples": float(len(dataset)),
        "inference_forward_calls": float(compute["forward_calls"]),
        "inference_token_positions": float(compute["token_positions"]),
        "approx_inference_flops": float(2 * model.parameter_count * compute["token_positions"]),
        "base_generation_forward_calls": float(base_calls),
        "base_generation_token_positions": float(base_positions),
        "approx_base_generation_flops": float(2 * model.parameter_count * base_positions),
        "validation_forward_calls": float(validation_calls),
        "validation_token_positions": float(validation_positions),
        "diagnostic_intervention_forward_calls": float(intervention_calls),
        "diagnostic_intervention_token_positions": float(intervention_positions),
        "inference_wall_seconds": elapsed,
        "generated_tokens": float(len(dataset) * (dataset.max_facets + 1)),
    }
    if zero_exact:
        metrics.update(
            zero_goal_exact_match=mean(zero_exact),
            shuffled_goal_exact_match=mean(shuffle_exact),
            zero_goal_effect=mean(exact) - mean(zero_exact),
            shuffled_goal_effect=mean(exact) - mean(shuffle_exact),
            same_goal_paraphrase_cosine=mean(paraphrase_similarity),
            different_goal_cosine=mean(different_goal_similarity),
            goal_separation_margin=(
                mean(paraphrase_similarity) - mean(different_goal_similarity)
            ),
            paraphrase_goal_substitution_exact_match=mean(paraphrase_goal_exact),
            facet_goal_substitution_success=mean(facet_substitution_success),
            no_mode_embedding_exact_match=mean(no_mode_exact),
            swapped_encode_generate_mode_exact_match=mean(swapped_mode_exact),
        )
    if labels:
        metrics.update(
            validator_auroc=binary_auroc(labels, scores),
            validator_auprc=average_precision(labels, scores),
            validator_brier=brier_score(labels, scores),
            validator_ece=expected_calibration_error(labels, scores),
            validator_ranking_accuracy=mean(rankings),
            validator_confidently_wrong_fpr=(
                sum(label == 0 and score >= 0.9 for label, score in zip(labels, scores))
                / sum(label == 0 for label in labels)
            ),
        )
    return metrics, predictions


def run(config: dict[str, Any], output_root: Path, repository: Path) -> Path:
    seed = int(config["seed"])
    baseline = str(config["baseline"])
    if baseline not in BASELINES:
        raise ValueError(f"Unknown baseline {baseline!r}")
    seed_everything(seed)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_root / str(config.get("experiment", "paper1")) / baseline / f"{stamp}-seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(run_dir / "config.json", config)
    provenance = capture_provenance(repository, config)

    train_data = _dataset(config, "train", seed)
    validation_data = _dataset(config, "validation", seed)
    test_data = _dataset(config, "test", seed)
    datasets = {
        "train": train_data.manifest(),
        "validation": validation_data.manifest(),
        "test": test_data.manifest(),
    }
    provenance["datasets"] = datasets
    provenance["dataset_sha256"] = stable_hash(datasets)
    atomic_write_json(run_dir / "provenance.json", provenance)

    device = _device(str(config.get("device", "auto")))
    model = _model(config, train_data).to(device)
    before = time.perf_counter()
    history, updates = train(model, train_data, config, device)
    train_compute = model.compute_stats()
    train_wall = time.perf_counter() - before
    validation = validation_losses(model, validation_data, config, device)
    metrics, predictions = evaluate(model, test_data, config, device)
    metrics.update(
        baseline=baseline,
        seed=seed,
        parameter_count=model.parameter_count,
        optimizer_updates=updates,
        training_wall_seconds=train_wall,
        training_forward_calls=train_compute["forward_calls"],
        training_token_positions=train_compute["token_positions"],
        approx_training_flops=6 * model.parameter_count * train_compute["token_positions"],
        device=str(device),
        **validation,
    )
    atomic_write_json(run_dir / "history.json", history)
    atomic_write_json(run_dir / "metrics.json", metrics)
    with (run_dir / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    torch.save(
        {"model": model.state_dict(), "config": config, "metrics": metrics},
        run_dir / "checkpoint.pt",
    )
    print(json.dumps({"run_dir": str(run_dir), **metrics}, sort_keys=True))
    return run_dir


def run_suite(config: dict[str, Any], output_root: Path, repository: Path) -> list[Path]:
    runs = []
    for baseline in config.get("baselines", BASELINES):
        for seed in config.get("seeds", [0, 1, 2]):
            resolved = deepcopy(config)
            resolved.pop("baselines", None)
            resolved.pop("seeds", None)
            resolved["baseline"] = baseline
            resolved["seed"] = seed
            runs.append(run(resolved, output_root, repository))
    summary: dict[str, dict[str, dict[str, float]]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in runs:
        grouped[path.parent.name].append(json.loads((path / "metrics.json").read_text()))
    for baseline, rows in grouped.items():
        summary[baseline] = {}
        numeric = sorted(
            key for key in set().union(*(row.keys() for row in rows))
            if all(isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool) for row in rows)
        )
        for key in numeric:
            values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
            summary[baseline][key] = {
                "count": len(values), "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
            }
    suite_dir = output_root / str(config.get("experiment", "paper1"))
    atomic_write_json(suite_dir / "latest-suite-summary.json", summary)
    return runs


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TypeError("Experiment config must be a YAML mapping")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train-eval", "suite"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.command == "suite":
        run_suite(config, arguments.output_root, arguments.repository)
    else:
        run(config, arguments.output_root, arguments.repository)


if __name__ == "__main__":
    main()
