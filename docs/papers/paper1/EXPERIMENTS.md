# Paper 1 experiment protocol

The executable scaffold implements the full B0--B10 ladder with one shared
parameterization. Baselines differ only in active computation, so parameter counts are
identical by construction. The controlled generator stores independent facet values,
late constraints, explicitly delimited instruction-like distractors, ordered targets,
and one-facet near misses.

## CPU smoke test

```powershell
python -m pip install -e .
pytest
modal-llm train-eval --config configs/paper1/smoke.yaml
modal-llm suite --config configs/paper1/smoke-suite.yaml
```

## Main model and five-seed baseline suite

```powershell
modal-llm train-eval --config configs/paper1/main.yaml
modal-llm suite --config configs/paper1/baselines.yaml
```

Each run creates a new immutable timestamped directory below `results/` with the
resolved config, Git/runtime provenance, dataset-generator hash, epoch history,
checkpoint, raw predictions, task metrics, validation calibration, forward-pass and
token counts, wall times, and zero/shuffled-goal interventions. The suite aggregates
numeric metrics by baseline. Raw run directories are never reused or overwritten.

The synthetic suite is a diagnostic first stage, not evidence for broad language-model
quality. Held-out templates and corruption families, executable code tasks, natural
instruction benchmarks, and larger pretrained backbones are required before headline
claims.
