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


def _task_exact(
    generated: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
    max_facets: int,
    out_end: int,
) -> torch.Tensor:
    token_ok = generated[:, :max_facets].eq(target[:, :max_facets])
    return (token_ok | ~active).all(1) & generated[:, -1].eq(out_end)


def _sequence_exact(
    generated: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    out_end: int,
) -> torch.Tensor:
    token_ok = generated.eq(target)
    return (token_ok | ~target_mask).all(1) & generated[:, -1].eq(out_end)


def _dataset(config: dict[str, Any], split: str, seed: int) -> ConstraintDataset:
    data = config["data"]
    size_key = {"train": "train_size", "validation": "validation_size", "test": "test_size"}[split]
    prompt_families = dict(data.get("prompt_families") or {})
    corruption_families = dict(data.get("corruption_families") or {})
    namespaces = dict(data.get("namespaces") or {})
    return ConstraintDataset(
        int(data[size_key]), seed=seed, split=split,
        max_facets=int(data.get("max_facets", 4)),
        min_facets=int(data.get("min_facets", 2)),
        max_distractors=int(data.get("max_distractors", 2)),
        max_filler=int(data.get("max_filler", 3)),
        prompt_family=prompt_families.get(split),
        corruption_family=corruption_families.get(split),
        goal_prompt_style=str(data.get("goal_prompt_style", "rendered")),
        direct_goal_exposure=float(data.get("direct_goal_exposure", 0.0)),
        target_repetitions=int(data.get("target_repetitions", 1)),
        namespace=namespaces.get(split),
    )


def _model(config: dict[str, Any], dataset: ConstraintDataset) -> ModeTransformer:
    values = config["model"]
    return ModeTransformer(
        dataset.vocab, str(config["baseline"]),
        d_model=int(values.get("d_model", 64)), nhead=int(values.get("nhead", 4)),
        layers=int(values.get("layers", 2)), ff_mult=int(values.get("ff_mult", 4)),
        dropout=float(values.get("dropout", 0.1)),
        max_length=int(values.get("max_length", 128)),
        generation_prompt_only=bool(values.get("generation_prompt_only", False)),
        goal_vectors=int(values.get("goal_vectors", 1)),
        goal_pooling=str(values.get("goal_pooling", "learned_queries")),
        z_injection_schedule=str(values.get("z_injection_schedule", "input_only")),
        z_injection_period=int(values.get("z_injection_period", 2)),
        conditioning_mode=str(values.get("conditioning_mode", "additive")),
        prefix_tokens=int(values.get("prefix_tokens", 4)),
    )


@torch.no_grad()
def _collect_goal_states(
    model: ModeTransformer,
    dataset: ConstraintDataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_examples)
    model.eval()
    states: list[torch.Tensor] = []
    bits: list[torch.Tensor] = []
    active: list[torch.Tensor] = []
    for raw in loader:
        batch = _move(raw, device)
        _, goal_prompt = model.prompt_channels(batch)
        _, goal = model.encode(goal_prompt)
        assert goal is not None
        states.append(goal.detach().cpu().reshape(goal.shape[0], -1))
        bits.append(batch["bits"].detach().cpu())
        active.append(batch["active_facets"].detach().cpu())
    return {
        "goal": torch.cat(states, dim=0),
        "bits": torch.cat(bits, dim=0),
        "active_facets": torch.cat(active, dim=0),
    }


def _probe_metrics(
    logits: torch.Tensor,
    bits: torch.Tensor,
    active: torch.Tensor,
) -> dict[str, float]:
    predictions = (logits.sigmoid() >= 0.5).to(bits.dtype)
    mask = active > 0
    correct = (predictions == bits) | ~mask
    denominator = float(mask.sum().item())
    facet_accuracy = float(((predictions == bits) & mask).sum().item() / denominator) if denominator else math.nan
    joint_exact = float(correct.all(dim=1).float().mean().item())
    return {
        "facet_accuracy": facet_accuracy,
        "joint_exact_match": joint_exact,
    }


def _train_goal_probe(
    train: dict[str, torch.Tensor],
    validation: dict[str, torch.Tensor],
    test: dict[str, torch.Tensor],
    *,
    hidden_width: int | None,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, float]:
    input_width = train["goal"].shape[1]
    output_width = train["bits"].shape[1]
    if hidden_width is None:
        probe = torch.nn.Linear(input_width, output_width)
    else:
        probe = torch.nn.Sequential(
            torch.nn.Linear(input_width, hidden_width),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_width, output_width),
        )
    probe = probe.to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_value = math.inf
    for _ in range(epochs):
        probe.train()
        optimizer.zero_grad(set_to_none=True)
        logits = probe(train["goal"].to(device))
        raw = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, train["bits"].to(device), reduction="none"
        )
        loss = (raw * train["active_facets"].to(device)).sum() / train["active_facets"].to(device).sum()
        loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            validation_logits = probe(validation["goal"].to(device))
            raw_validation = torch.nn.functional.binary_cross_entropy_with_logits(
                validation_logits, validation["bits"].to(device), reduction="none"
            )
            validation_loss = (
                (raw_validation * validation["active_facets"].to(device)).sum()
                / validation["active_facets"].to(device).sum()
            )
        if float(validation_loss) < best_value:
            best_value = float(validation_loss)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in probe.state_dict().items()
            }
    assert best_state is not None
    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        train_logits = probe(train["goal"].to(device)).cpu()
        validation_logits = probe(validation["goal"].to(device)).cpu()
        test_logits = probe(test["goal"].to(device)).cpu()
    metrics = {}
    for split_name, logits, values in (
        ("train", train_logits, train),
        ("validation", validation_logits, validation),
        ("test", test_logits, test),
    ):
        for metric_name, metric_value in _probe_metrics(
            logits, values["bits"], values["active_facets"]
        ).items():
            metrics[f"{split_name}_{metric_name}"] = metric_value
    return metrics


def train(
    model: ModeTransformer,
    dataset: ConstraintDataset,
    validation_dataset: ConstraintDataset,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
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
    history: list[dict[str, Any]] = []
    updates = 0
    selection_validation_calls = 0
    selection_validation_positions = 0
    model.reset_compute_stats()
    model.train()
    warmup_epochs = int(values.get("goal_warmup_epochs", 0)) if model.spec.use_goal else 0
    invariance_weight = float(weights.get("goal_invariance", 0.0))
    for epoch in range(warmup_epochs):
        totals: dict[str, list[float]] = defaultdict(list)
        started = time.perf_counter()
        for raw in loader:
            batch = _move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            losses = model.goal_objective(batch, invariance_weight=invariance_weight)
            losses["total"].backward()
            clip_grad_norm_(model.parameters(), float(values.get("grad_clip", 1.0)))
            optimizer.step()
            updates += 1
            for key, value in losses.items():
                totals[key].append(float(value.detach()))
        row: dict[str, Any] = {
            f"train_{key}": mean(items) for key, items in totals.items()
        }
        row.update(
            stage="goal_warmup",
            epoch=epoch + 1,
            updates=updates,
            wall_seconds=time.perf_counter() - started,
        )
        history.append(row)
        log_every = int(values.get("log_every", 1))
        if (epoch + 1) % log_every == 0 or epoch + 1 == warmup_epochs:
            print(json.dumps(row, sort_keys=True))

    best_metric = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_validations = 0
    validation_every = int(values.get("validation_every", 1))
    patience = int(values.get("early_stopping_patience", 0))
    completed_joint_epochs = 0
    for epoch in range(int(values.get("epochs", 5))):
        totals: dict[str, list[float]] = defaultdict(list)
        started = time.perf_counter()
        model.train()
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
        row.update(
            stage="joint",
            epoch=epoch + 1,
            updates=updates,
            wall_seconds=time.perf_counter() - started,
        )
        completed_joint_epochs = epoch + 1
        if (epoch + 1) % validation_every == 0:
            before_validation = model.compute_stats()
            held_out = validation_losses(model, validation_dataset, config, device)
            after_validation = model.compute_stats()
            selection_validation_calls += (
                after_validation["forward_calls"] - before_validation["forward_calls"]
            )
            selection_validation_positions += (
                after_validation["token_positions"] - before_validation["token_positions"]
            )
            row.update(held_out)
            selected = float(held_out["validation_total"])
            if selected < best_metric:
                best_metric = selected
                best_epoch = epoch + 1
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                stale_validations = 0
            else:
                stale_validations += 1
            model.train()
        history.append(row)
        log_every = int(values.get("log_every", 1))
        if (epoch + 1) % log_every == 0 or epoch + 1 == int(values.get("epochs", 5)):
            print(json.dumps(row, sort_keys=True))
        if patience and stale_validations >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    compute = model.compute_stats()
    training_compute = {
        "forward_calls": compute["forward_calls"] - selection_validation_calls,
        "token_positions": compute["token_positions"] - selection_validation_positions,
        "selection_validation_forward_calls": selection_validation_calls,
        "selection_validation_token_positions": selection_validation_positions,
        "best_epoch": best_epoch or completed_joint_epochs,
        "completed_joint_epochs": completed_joint_epochs,
        "goal_warmup_epochs": warmup_epochs,
    }
    return history, updates, training_compute


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


def evaluate_goal_probes(
    model: ModeTransformer,
    train_dataset: ConstraintDataset,
    validation_dataset: ConstraintDataset,
    test_dataset: ConstraintDataset,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    if not model.spec.use_goal:
        return {}
    probe_config = (
        config.get("evaluation", {}).get("goal_probe")
        or {"enabled": True, "epochs": 200, "learning_rate": 1e-2, "weight_decay": 1e-4}
    )
    if not probe_config.get("enabled", True):
        return {}
    batch_size = int(probe_config.get("batch_size", config["evaluation"].get("batch_size", 64)))
    train_states = _collect_goal_states(model, train_dataset, batch_size, device)
    validation_states = _collect_goal_states(model, validation_dataset, batch_size, device)
    test_states = _collect_goal_states(model, test_dataset, batch_size, device)
    common = {
        "device": device,
        "epochs": int(probe_config.get("epochs", 200)),
        "learning_rate": float(probe_config.get("learning_rate", 1e-2)),
        "weight_decay": float(probe_config.get("weight_decay", 1e-4)),
    }
    linear = _train_goal_probe(
        train_states, validation_states, test_states, hidden_width=None, **common
    )
    mlp = _train_goal_probe(
        train_states,
        validation_states,
        test_states,
        hidden_width=int(probe_config.get("mlp_hidden", 64)),
        **common,
    )
    metrics = {}
    for prefix, values in (("goal_probe_linear", linear), ("goal_probe_mlp", mlp)):
        for name, value in values.items():
            metrics[f"{prefix}_{name}"] = value
    return metrics


@torch.no_grad()
def evaluate_horizon(
    model: ModeTransformer,
    dataset: ConstraintDataset,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate long scheduled outputs with only the required shuffled-Z control."""

    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluation"].get("batch_size", 16)),
        shuffle=False,
        collate_fn=collate_examples,
    )
    model.eval()
    exact: list[float] = []
    shuffled_exact: list[float] = []
    satisfaction: list[float] = []
    shuffled_satisfaction: list[float] = []
    slot_swap_exact: list[float] = []
    slot_swap_selected: list[float] = []
    slot_swap_untouched: list[float] = []
    quartiles: list[list[float]] = [[] for _ in range(4)]
    shuffled_quartiles: list[list[float]] = [[] for _ in range(4)]
    predictions: list[dict[str, Any]] = []
    base_calls = base_positions = 0
    intervention_calls = intervention_positions = 0
    model.reset_compute_stats()
    started = time.perf_counter()
    for raw in loader:
        batch = _move(raw, device)
        generation_prompt, goal_prompt = model.prompt_channels(batch)
        before_base = model.compute_stats()
        generated, goal, _ = model.generate(
            generation_prompt, batch["target"].shape[1], goal_prompt=goal_prompt
        )
        after_base = model.compute_stats()
        base_calls += after_base["forward_calls"] - before_base["forward_calls"]
        base_positions += after_base["token_positions"] - before_base["token_positions"]
        shuffled_generated = None
        slot_swapped_generated = None
        if goal is not None:
            shuffled_generated, _, _ = model.generate(
                generation_prompt,
                batch["target"].shape[1],
                forced_goal=goal.roll(1, 0),
            )
            after_intervention = model.compute_stats()
            intervention_calls += (
                after_intervention["forward_calls"] - after_base["forward_calls"]
            )
            intervention_positions += (
                after_intervention["token_positions"] - after_base["token_positions"]
            )
            if model.goal_pooling == "facet_tokens":
                _, counterfactual_goal = model.encode(
                    model.counterfactual_goal_channel(batch)
                )
                swapped_goal = goal.clone()
                row_indices = torch.arange(len(goal), device=device)
                swapped_goal[row_indices, batch["corrupt_facet"]] = counterfactual_goal[
                    row_indices, batch["corrupt_facet"]
                ]
                slot_swapped_generated, _, _ = model.generate(
                    generation_prompt,
                    batch["target"].shape[1],
                    forced_goal=swapped_goal,
                )
                after_slot_swap = model.compute_stats()
                intervention_calls += (
                    after_slot_swap["forward_calls"]
                    - after_intervention["forward_calls"]
                )
                intervention_positions += (
                    after_slot_swap["token_positions"]
                    - after_intervention["token_positions"]
                )

        target_mask = batch["target_mask"]
        correct = generated.eq(batch["target"]) & target_mask
        denominator = target_mask.sum(1).clamp_min(1)
        batch_satisfaction = correct.sum(1).float() / denominator
        batch_exact = _sequence_exact(
            generated, batch["target"], target_mask, model.vocab.OUT_END
        )
        exact.extend(batch_exact.float().cpu().tolist())
        satisfaction.extend(batch_satisfaction.cpu().tolist())
        batch_shuffled_satisfaction = None
        batch_shuffled_exact = None
        if shuffled_generated is not None:
            shuffled_correct = shuffled_generated.eq(batch["target"]) & target_mask
            batch_shuffled_satisfaction = shuffled_correct.sum(1).float() / denominator
            batch_shuffled_exact = _sequence_exact(
                shuffled_generated, batch["target"], target_mask, model.vocab.OUT_END
            )
            shuffled_exact.extend(batch_shuffled_exact.float().cpu().tolist())
            shuffled_satisfaction.extend(batch_shuffled_satisfaction.cpu().tolist())
        batch_slot_swap_exact = None
        batch_slot_swap_selected = None
        batch_slot_swap_untouched = None
        if slot_swapped_generated is not None:
            selected_mask = batch["target_facets"].eq(
                batch["corrupt_facet"].unsqueeze(1)
            )
            untouched_mask = target_mask & ~selected_mask
            batch_slot_swap_selected = (
                slot_swapped_generated.eq(batch["counterfactual_target"])
                .logical_and(selected_mask)
                .sum(1)
                .float()
                / selected_mask.sum(1).clamp_min(1)
            )
            batch_slot_swap_untouched = (
                slot_swapped_generated.eq(batch["target"])
                .logical_and(untouched_mask)
                .sum(1)
                .float()
                / untouched_mask.sum(1).clamp_min(1)
            )
            batch_slot_swap_exact = _sequence_exact(
                slot_swapped_generated,
                batch["counterfactual_target"],
                target_mask,
                model.vocab.OUT_END,
            )
            slot_swap_selected.extend(batch_slot_swap_selected.cpu().tolist())
            slot_swap_untouched.extend(batch_slot_swap_untouched.cpu().tolist())
            slot_swap_exact.extend(batch_slot_swap_exact.float().cpu().tolist())

        active_positions = target_mask[0].nonzero().flatten()
        for quartile, positions in enumerate(torch.tensor_split(active_positions, 4)):
            quartiles[quartile].extend(
                generated[:, positions]
                .eq(batch["target"][:, positions])
                .float()
                .mean(1)
                .cpu()
                .tolist()
            )
            if shuffled_generated is not None:
                shuffled_quartiles[quartile].extend(
                    shuffled_generated[:, positions]
                    .eq(batch["target"][:, positions])
                    .float()
                    .mean(1)
                    .cpu()
                    .tolist()
                )

        for index, example_id in enumerate(raw["ids"]):
            row = {
                "id": example_id,
                "target": raw["target"][index].tolist(),
                "generated": generated[index].cpu().tolist(),
                "satisfaction": float(batch_satisfaction[index]),
                "exact": bool(batch_exact[index]),
            }
            if shuffled_generated is not None:
                assert batch_shuffled_satisfaction is not None
                assert batch_shuffled_exact is not None
                row.update(
                    shuffled_generated=shuffled_generated[index].cpu().tolist(),
                    shuffled_satisfaction=float(batch_shuffled_satisfaction[index]),
                    shuffled_exact=bool(batch_shuffled_exact[index]),
                )
            if slot_swapped_generated is not None:
                assert batch_slot_swap_exact is not None
                assert batch_slot_swap_selected is not None
                assert batch_slot_swap_untouched is not None
                row.update(
                    counterfactual_target=raw["counterfactual_target"][index].tolist(),
                    facet_slot_swapped_generated=slot_swapped_generated[index]
                    .cpu()
                    .tolist(),
                    facet_slot_swap_exact=bool(batch_slot_swap_exact[index]),
                    facet_slot_swap_selected_satisfaction=float(
                        batch_slot_swap_selected[index]
                    ),
                    facet_slot_swap_untouched_satisfaction=float(
                        batch_slot_swap_untouched[index]
                    ),
                )
            predictions.append(row)

    elapsed = time.perf_counter() - started
    compute = model.compute_stats()
    metrics = {
        "task_exact_match": mean(exact),
        "task_facet_satisfaction": mean(satisfaction),
        "goal_drift": mean(quartiles[0]) - mean(quartiles[3]),
        "evaluation_examples": float(len(dataset)),
        "target_output_length": float(dataset.max_facets * dataset.target_repetitions + 1),
        "generated_tokens": float(
            len(dataset) * (dataset.max_facets * dataset.target_repetitions + 1)
        ),
        "inference_forward_calls": float(compute["forward_calls"]),
        "inference_token_positions": float(compute["token_positions"]),
        "approx_inference_flops": float(
            2 * model.active_parameter_count * compute["token_positions"]
        ),
        "base_generation_forward_calls": float(base_calls),
        "base_generation_token_positions": float(base_positions),
        "approx_base_generation_flops": float(
            2 * model.active_parameter_count * base_positions
        ),
        "diagnostic_intervention_forward_calls": float(intervention_calls),
        "diagnostic_intervention_token_positions": float(intervention_positions),
        "validation_forward_calls": 0.0,
        "validation_token_positions": 0.0,
        "inference_wall_seconds": elapsed,
    }
    for index in range(4):
        metrics[f"task_quartile_{index + 1}_satisfaction"] = mean(quartiles[index])
    if shuffled_exact:
        metrics.update(
            shuffled_goal_exact_match=mean(shuffled_exact),
            shuffled_goal_effect=mean(exact) - mean(shuffled_exact),
            shuffled_goal_facet_satisfaction=mean(shuffled_satisfaction),
            shuffled_goal_facet_effect=mean(satisfaction) - mean(
                shuffled_satisfaction
            ),
            shuffled_goal_drift=mean(shuffled_quartiles[0])
            - mean(shuffled_quartiles[3]),
        )
        for index in range(4):
            metrics[f"shuffled_goal_quartile_{index + 1}_satisfaction"] = mean(
                shuffled_quartiles[index]
            )
    if slot_swap_exact:
        metrics.update(
            facet_slot_swap_exact_match=mean(slot_swap_exact),
            facet_slot_swap_selected_satisfaction=mean(slot_swap_selected),
            facet_slot_swap_untouched_satisfaction=mean(slot_swap_untouched),
        )
    return metrics, predictions


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
    corruption_probabilities: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"positive": [], "negative": [], "ranking": []}
    )
    zero_exact: list[float] = []
    shuffle_exact: list[float] = []
    constant_exact: list[float] = []
    random_exact: list[float] = []
    paraphrase_goal_exact: list[float] = []
    facet_substitution_success: list[float] = []
    facet_substitution_isolated_success: list[float] = []
    counterfactual_exact: list[float] = []
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
    constant_goal: torch.Tensor | None = None
    for raw in loader:
        batch = _move(raw, device)
        generation_prompt, goal_prompt = model.prompt_channels(batch)
        before_base = model.compute_stats()
        generated, goal, _ = model.generate(
            generation_prompt, batch["target"].shape[1], goal_prompt=goal_prompt
        )
        after_base = model.compute_stats()
        base_calls += after_base["forward_calls"] - before_base["forward_calls"]
        base_positions += after_base["token_positions"] - before_base["token_positions"]
        active = batch["active_facets"].bool()
        token_ok = generated[:, : model.vocab.max_facets].eq(
            batch["target"][:, : model.vocab.max_facets]
        )
        facet_fraction = (token_ok & active).sum(1).float() / active.sum(1)
        batch_exact = _task_exact(
            generated,
            batch["target"],
            active,
            model.vocab.max_facets,
            model.vocab.OUT_END,
        )
        exact.extend(batch_exact.float().cpu().tolist())
        facets.extend(facet_fraction.cpu().tolist())
        batch_interventions: dict[str, torch.Tensor] = {}

        if model.spec.use_goal and goal is not None:
            zeros = torch.zeros_like(goal)
            zero_generated, _, zero_calls = model.generate(
                generation_prompt, batch["target"].shape[1], forced_goal=zeros
            )
            shuffled = goal.roll(1, 0)
            shuffled_generated, _, shuffle_calls = model.generate(
                generation_prompt, batch["target"].shape[1], forced_goal=shuffled
            )
            if constant_goal is None:
                constant_goal = goal[:1].detach().clone()
            constant = constant_goal.expand_as(goal)
            constant_generated, _, _ = model.generate(
                generation_prompt, batch["target"].shape[1], forced_goal=constant
            )
            noise_scale = goal.std(dim=0, unbiased=False).mean().clamp_min(1e-3)
            random_goal = torch.randn_like(goal) * noise_scale
            random_generated, _, _ = model.generate(
                generation_prompt, batch["target"].shape[1], forced_goal=random_goal
            )
            for values, destination in (
                (zero_generated, zero_exact),
                (shuffled_generated, shuffle_exact),
                (constant_generated, constant_exact),
                (random_generated, random_exact),
            ):
                destination.extend(
                    _task_exact(
                        values,
                        batch["target"],
                        active,
                        model.vocab.max_facets,
                        model.vocab.OUT_END,
                    )
                    .float()
                    .cpu()
                    .tolist()
                )
            paraphrase_goals = model.paraphrase_goal_channel(batch)
            _, paraphrase_goal = model.encode(paraphrase_goals)
            goal_flat = goal.reshape(goal.shape[0], -1)
            paraphrase_flat = paraphrase_goal.reshape(paraphrase_goal.shape[0], -1)
            paraphrase_similarity.extend(
                torch.nn.functional.cosine_similarity(goal_flat, paraphrase_flat).cpu().tolist()
            )
            if len(goal) > 1:
                different_goal_similarity.extend(
                    torch.nn.functional.cosine_similarity(
                        goal_flat, goal_flat.roll(1, 0)
                    ).cpu().tolist()
                )
            paraphrase_generated, _, _ = model.generate(
                generation_prompt, batch["target"].shape[1], forced_goal=paraphrase_goal
            )
            paraphrase_goal_exact.extend(
                _task_exact(
                    paraphrase_generated,
                    batch["target"],
                    active,
                    model.vocab.max_facets,
                    model.vocab.OUT_END,
                )
                .float()
                .cpu()
                .tolist()
            )

            _, counterfactual_goal = model.encode(model.counterfactual_goal_channel(batch))
            substituted, _, _ = model.generate(
                generation_prompt, batch["target"].shape[1], forced_goal=counterfactual_goal
            )
            row_index = torch.arange(len(substituted), device=device)
            changed = substituted[
                row_index, batch["corrupt_facet"]
            ].eq(batch["counterfactual_target"][row_index, batch["corrupt_facet"]])
            facet_substitution_success.extend(changed.float().cpu().tolist())
            untouched = active.clone()
            untouched[row_index, batch["corrupt_facet"]] = False
            untouched_correct = (
                substituted[:, : model.vocab.max_facets].eq(
                    batch["target"][:, : model.vocab.max_facets]
                )
                | ~untouched
            ).all(1)
            isolated = (
                changed
                & untouched_correct
                & substituted[:, -1].eq(model.vocab.OUT_END)
            )
            facet_substitution_isolated_success.extend(isolated.float().cpu().tolist())
            counterfactual_exact.extend(
                _task_exact(
                    substituted,
                    batch["counterfactual_target"],
                    active,
                    model.vocab.max_facets,
                    model.vocab.OUT_END,
                )
                .float()
                .cpu()
                .tolist()
            )

            mode_weights = model.mode_embedding.weight.detach().clone()
            model.mode_embedding.weight.zero_()
            without_mode, _, _ = model.generate(
                generation_prompt, batch["target"].shape[1], goal_prompt=goal_prompt
            )
            model.mode_embedding.weight.copy_(mode_weights)
            no_mode_exact.extend(
                _task_exact(
                    without_mode,
                    batch["target"],
                    active,
                    model.vocab.max_facets,
                    model.vocab.OUT_END,
                )
                .float()
                .cpu()
                .tolist()
            )

            swapped = mode_weights.clone()
            swapped[[0, 1]] = swapped[[1, 0]]
            model.mode_embedding.weight.copy_(swapped)
            swapped_mode, _, _ = model.generate(
                generation_prompt, batch["target"].shape[1], goal_prompt=goal_prompt
            )
            model.mode_embedding.weight.copy_(mode_weights)
            swapped_mode_exact.extend(
                _task_exact(
                    swapped_mode,
                    batch["target"],
                    active,
                    model.vocab.max_facets,
                    model.vocab.OUT_END,
                )
                .float()
                .cpu()
                .tolist()
            )
            batch_interventions = {
                "zero_generated": zero_generated,
                "shuffled_generated": shuffled_generated,
                "constant_generated": constant_generated,
                "random_generated": random_generated,
                "paraphrase_generated": paraphrase_generated,
                "counterfactual_generated": substituted,
            }

        after_interventions = model.compute_stats()
        intervention_calls += after_interventions["forward_calls"] - after_base["forward_calls"]
        intervention_positions += (
            after_interventions["token_positions"] - after_base["token_positions"]
        )

        positive_probability = negative_probability = None
        if model.spec.validator is not None:
            positive, _ = model.validation_logits(
                batch["prompt"], batch["target"], goal, goal_prompt=goal_prompt
            )
            negative, _ = model.validation_logits(
                batch["prompt"], batch["corrupted"], goal, goal_prompt=goal_prompt
            )
            positive_probability = positive.sigmoid()
            negative_probability = negative.sigmoid()
            labels.extend([1] * len(positive) + [0] * len(negative))
            scores.extend(positive_probability.cpu().tolist() + negative_probability.cpu().tolist())
            rankings.extend((positive > negative).float().cpu().tolist())
            for index, corruption_type in enumerate(raw["corruption_types"]):
                group = corruption_probabilities[corruption_type]
                group["positive"].append(float(positive_probability[index]))
                group["negative"].append(float(negative_probability[index]))
                group["ranking"].append(float(positive[index] > negative[index]))

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
                "corruption_type": raw["corruption_types"][index],
            }
            if batch_interventions:
                row.update(
                    {
                        name: values[index].cpu().tolist()
                        for name, values in batch_interventions.items()
                    },
                    counterfactual_target=raw["counterfactual_target"][index].tolist(),
                    counterfactual_facet=int(raw["corrupt_facet"][index]),
                )
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
        "approx_inference_flops": float(
            2 * model.active_parameter_count * compute["token_positions"]
        ),
        "base_generation_forward_calls": float(base_calls),
        "base_generation_token_positions": float(base_positions),
        "approx_base_generation_flops": float(
            2 * model.active_parameter_count * base_positions
        ),
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
            constant_goal_exact_match=mean(constant_exact),
            random_goal_exact_match=mean(random_exact),
            zero_goal_effect=mean(exact) - mean(zero_exact),
            shuffled_goal_effect=mean(exact) - mean(shuffle_exact),
            constant_goal_effect=mean(exact) - mean(constant_exact),
            random_goal_effect=mean(exact) - mean(random_exact),
            correct_minus_shuffled_goal=mean(exact) - mean(shuffle_exact),
            same_goal_paraphrase_cosine=mean(paraphrase_similarity),
            different_goal_cosine=mean(different_goal_similarity),
            goal_separation_margin=(
                mean(paraphrase_similarity) - mean(different_goal_similarity)
            ),
            paraphrase_goal_substitution_exact_match=mean(paraphrase_goal_exact),
            facet_goal_substitution_success=mean(facet_substitution_success),
            facet_goal_substitution_isolated_success=mean(
                facet_substitution_isolated_success
            ),
            counterfactual_goal_exact_match=mean(counterfactual_exact),
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
        for corruption_type, group in sorted(corruption_probabilities.items()):
            family_labels = [1] * len(group["positive"]) + [0] * len(group["negative"])
            family_scores = group["positive"] + group["negative"]
            safe_name = corruption_type.replace("-", "_")
            metrics[f"validator_{safe_name}_auroc"] = binary_auroc(
                family_labels, family_scores
            )
            metrics[f"validator_{safe_name}_ranking_accuracy"] = mean(group["ranking"])
    return metrics, predictions


@torch.no_grad()
def verify_checkpoint_reload(
    checkpoint_path: Path,
    model: ModeTransformer,
    dataset: ConstraintDataset,
    config: dict[str, Any],
    device: torch.device,
) -> bool:
    """Reload a checkpoint into a fresh model and compare deterministic predictions."""

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    reloaded = _model(config, dataset).to(device)
    reloaded.load_state_dict(payload["model"], strict=True)
    model.eval()
    reloaded.eval()
    sample_count = min(8, len(dataset))
    raw = collate_examples([dataset[index] for index in range(sample_count)])
    batch = _move(raw, device)
    generation_prompt, goal_prompt = model.prompt_channels(batch)
    expected, _, _ = model.generate(
        generation_prompt, batch["target"].shape[1], goal_prompt=goal_prompt
    )
    actual, _, _ = reloaded.generate(
        generation_prompt, batch["target"].shape[1], goal_prompt=goal_prompt
    )
    states_equal = all(
        torch.equal(value, reloaded.state_dict()[name])
        for name, value in model.state_dict().items()
    )
    return states_equal and torch.equal(expected, actual)


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
        split: {**dataset.manifest(), "content_sha256": dataset.content_hash()}
        for split, dataset in (
            ("train", train_data),
            ("validation", validation_data),
            ("test", test_data),
        )
    }
    provenance["datasets"] = datasets
    provenance["dataset_sha256"] = stable_hash(datasets)
    atomic_write_json(run_dir / "provenance.json", provenance)

    device = _device(str(config.get("device", "auto")))
    model = _model(config, train_data).to(device)
    before = time.perf_counter()
    history, updates, train_compute = train(
        model, train_data, validation_data, config, device
    )
    train_wall = time.perf_counter() - before
    validation = validation_losses(model, validation_data, config, device)
    if test_data.target_repetitions > 1:
        metrics, predictions = evaluate_horizon(model, test_data, config, device)
    else:
        metrics, predictions = evaluate(model, test_data, config, device)
    metrics.update(
        evaluate_goal_probes(model, train_data, validation_data, test_data, config, device)
    )
    metrics.update(
        baseline=baseline,
        seed=seed,
        parameter_count=model.parameter_count,
        active_parameter_count=model.active_parameter_count,
        optimizer_updates=updates,
        training_wall_seconds=train_wall,
        training_forward_calls=train_compute["forward_calls"],
        training_token_positions=train_compute["token_positions"],
        selection_validation_forward_calls=train_compute[
            "selection_validation_forward_calls"
        ],
        selection_validation_token_positions=train_compute[
            "selection_validation_token_positions"
        ],
        best_epoch=train_compute["best_epoch"],
        completed_joint_epochs=train_compute["completed_joint_epochs"],
        goal_warmup_epochs=train_compute["goal_warmup_epochs"],
        approx_training_flops=(
            6 * model.active_parameter_count * train_compute["token_positions"]
        ),
        device=str(device),
        **validation,
    )
    atomic_write_json(run_dir / "history.json", history)
    with (run_dir / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(
        {"model": model.state_dict(), "config": config, "metrics": metrics},
        checkpoint_path,
    )
    metrics["checkpoint_reload_verified"] = verify_checkpoint_reload(
        checkpoint_path, model, test_data, config, device
    )
    atomic_write_json(run_dir / "metrics.json", metrics)
    for metric, minimum in (config.get("assertions", {}).get("minimum", {}) or {}).items():
        if float(metrics[metric]) < float(minimum):
            raise AssertionError(
                f"Sanity threshold failed: {metric}={metrics[metric]} < {minimum}; "
                f"artifacts retained in {run_dir}"
            )
    for metric, maximum in (config.get("assertions", {}).get("maximum", {}) or {}).items():
        if float(metrics[metric]) > float(maximum):
            raise AssertionError(
                f"Sanity threshold failed: {metric}={metrics[metric]} > {maximum}; "
                f"artifacts retained in {run_dir}"
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
            if not values:
                continue
            summary[baseline][key] = {
                "count": len(values), "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                "ci95_half_width": (
                    1.96 * stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
                ),
            }
    suite_dir = output_root / str(config.get("experiment", "paper1"))
    comparisons: dict[str, dict[str, dict[str, float]]] = {}
    rows_by_baseline_seed = {
        baseline: {int(row["seed"]): row for row in rows}
        for baseline, rows in grouped.items()
    }
    requested_comparisons = config.get("comparisons") or [
        ["B3", "B2"], ["B5", "B3"], ["B5", "B4"],
        ["B8", "B7"], ["B10", "B9"],
    ]
    for treatment, control in requested_comparisons:
        if treatment not in rows_by_baseline_seed or control not in rows_by_baseline_seed:
            continue
        common_seeds = sorted(
            set(rows_by_baseline_seed[treatment]) & set(rows_by_baseline_seed[control])
        )
        comparison_metrics: dict[str, dict[str, float]] = {}
        for metric in sorted(
            set.intersection(
                *(set(rows_by_baseline_seed[baseline][seed]) for baseline in (treatment, control)
                  for seed in common_seeds)
            ) if common_seeds else set()
        ):
            differences = []
            for seed in common_seeds:
                left = rows_by_baseline_seed[treatment][seed].get(metric)
                right = rows_by_baseline_seed[control][seed].get(metric)
                if (
                    isinstance(left, (int, float)) and not isinstance(left, bool)
                    and isinstance(right, (int, float)) and not isinstance(right, bool)
                    and math.isfinite(float(left)) and math.isfinite(float(right))
                ):
                    differences.append(float(left) - float(right))
            if differences:
                comparison_metrics[metric] = {
                    "count": len(differences),
                    "mean_difference": mean(differences),
                    "std_difference": stdev(differences) if len(differences) > 1 else 0.0,
                    "ci95_half_width": (
                        1.96 * stdev(differences) / math.sqrt(len(differences))
                        if len(differences) > 1 else 0.0
                    ),
                }
        comparisons[f"{treatment}-minus-{control}"] = comparison_metrics
    suite_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    suite_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": stable_hash(config),
        "runs": [str(path) for path in runs],
        "groups": summary,
        "paired_comparisons": comparisons,
    }
    atomic_write_json(suite_dir / f"suite-{suite_stamp}.json", suite_payload)
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
