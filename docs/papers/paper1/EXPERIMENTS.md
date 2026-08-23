# Paper 1 experiment protocol

The executable scaffold implements the full B0--B10 ladder with one shared
parameterization. Baselines differ only in active computation, so parameter counts are
identical by construction; total and active parameter counts are both reported. The
controlled generator stores independent facet values, late constraints, explicitly
delimited instruction-like distractors, ordered targets, counterfactual goals, and
machine-checkable corruptions.

## CPU smoke test

```powershell
python -m pip install -e .
pytest
modal-llm train-eval --config configs/paper1/smoke.yaml
modal-llm suite --config configs/paper1/smoke-suite.yaml
```

## Convergence and held-out pilots

```powershell
modal-llm train-eval --config configs/paper1/overfit.yaml
modal-llm suite --config configs/paper1/overfit-suite.yaml
modal-llm suite --config configs/paper1/pilot-suite.yaml
modal-llm suite --config configs/paper1/pilot-staged-suite.yaml
```

The overfit configurations deliberately share example namespaces and assert at least
95% exact match. They are implementation gates, not generalization measurements. The
pilot configurations use disjoint examples and prompt topologies: standard training,
reordered validation, and interleaved test prompts. Validation uses late-facet flips;
test evaluation mixes single/double flips, truncation, and invalid end markers. The
staged pilot adds goal/facet warmup, paraphrase invariance, validation-selected
checkpoints, and bounded early stopping.

The checked-in source data for the current one-seed implementation diagnostic is
[`results/pilot_iteration2.json`](results/pilot_iteration2.json). It records both the
shared-example memorization gate and the genuinely held-out pilot. These numbers are
used to reject weak training configurations, not as evidence for the paper's headline
hypotheses.

## Next iteration: Z-only causal gate

```powershell
modal-llm train-eval --config configs/paper1/z-only-gate.yaml
modal-llm train-eval --config configs/paper1/z-only-multivector.yaml
modal-llm train-eval --config configs/paper1/z-only-all-layers.yaml
modal-llm train-eval --config configs/paper1/z-only-periodic.yaml
modal-llm train-eval --config configs/paper1/z-only-late-layers.yaml
```

This configuration removes token-level goal access from the generator by using the
separate `generation_prompt` content channel together with `goal_prompt` encoding and
latched `Z_G` conditioning. The run reports correct-, shuffled-, zero-, constant-, and
random-goal interventions, plus frozen-goal linear and MLP probes over held-out facet
labels. Treat this as the first go/no-go gate for the next iteration: do not start the
depth/time sweep or the five-seed baseline suite unless correct-$Z$ clearly beats
shuffled-$Z$ on held-out tasks and the probes show strong goal decodability.

The multi-vector variant is the first Paper 1 context/prefix extension under the same
Z-only protocol. It keeps additive latent conditioning but replaces the single goal
vector with a learned bank of latent goal vectors. Use it to test whether one-shot
single-vector compression is the current bottleneck before moving on to deeper
architectural sweeps.

The all-layers variant is the first direct persistence diagnostic. It re-injects the
 same latched goal conditioning before every shared transformer block during generation.
 Compare it against the default input-only conditioning to test whether persistent
goal availability across depth matters more than one-shot compression at the input.

The periodic variant is the next selective persistence control. It re-injects the same
latched goal condition every $K$ layers, with $K=2$ in the first pilot, to test whether
less intrusive repeated access can recover causal benefits without the interference seen
under all-layer additive reinjection.

The late-layer variant is the follow-up top-down-control diagnostic. It re-injects the
same latent only in the final quarter of generation layers, asking whether a late
goal reminder is more useful than early or repeated additive conditioning in this small
shared-weight setting.

## Main model and five-seed baseline suite

```powershell
modal-llm train-eval --config configs/paper1/main.yaml
modal-llm suite --config configs/paper1/baselines.yaml
```

Each run creates a new immutable timestamped directory below `results/` with the
resolved config, Git/runtime provenance, hashes of the fully materialized datasets,
epoch history, checkpoint, checkpoint-reload verification, raw predictions, task
metrics, per-corruption validation metrics, forward-pass and token counts, wall times,
and causal interventions. Goal encoding is computed once and latched during decoding.
Suite summaries are timestamped and immutable; they include normal-approximation 95%
confidence intervals and seed-paired baseline differences. Raw run directories are
never reused or overwritten.

The synthetic suite is a diagnostic first stage, not evidence for broad language-model
quality. The held-out synthetic families diagnose architectural and optimization
failures. Executable transformations/code tasks, natural instruction benchmarks, and
larger pretrained backbones are still required before headline claims.
