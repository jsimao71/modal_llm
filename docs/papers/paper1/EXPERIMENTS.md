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
modal-llm train-eval --config configs/paper1/text-prefix.yaml
modal-llm train-eval --config configs/paper1/text-prefix-converged.yaml
modal-llm train-eval --config configs/paper1/z-only-gate.yaml
modal-llm train-eval --config configs/paper1/z-only-multivector.yaml
modal-llm train-eval --config configs/paper1/z-only-all-layers.yaml
modal-llm train-eval --config configs/paper1/z-only-periodic.yaml
modal-llm train-eval --config configs/paper1/z-only-late-layers.yaml
modal-llm train-eval --config configs/paper1/z-only-late-layers-converged.yaml
modal-llm train-eval --config configs/paper1/z-only-canonical-goal-converged.yaml
modal-llm train-eval --config configs/paper1/z-direct-25.yaml
modal-llm train-eval --config configs/paper1/z-direct-50.yaml
modal-llm train-eval --config configs/paper1/z-direct-75.yaml
modal-llm train-eval --config configs/paper1/z-direct-100.yaml
modal-llm train-eval --config configs/paper1/horizon-32-z.yaml
modal-llm train-eval --config configs/paper1/horizon-32-direct.yaml
modal-llm train-eval --config configs/paper1/z-only-multivector-late-layers.yaml
modal-llm train-eval --config configs/paper1/z-only-prefix.yaml
modal-llm train-eval --config configs/paper1/z-only-prefix-2.yaml
modal-llm train-eval --config configs/paper1/z-only-prefix-8.yaml
modal-llm train-eval --config configs/paper1/z-only-prefix-kv.yaml
```

The text-prefix configuration is the matched direct-channel control: its causal
generator receives the complete rendered task prompt, including ordinary goal tokens,
and no latent goal state. Configurations with `generation_prompt_only: true` instead
remove token-level goal access from the generator by using the
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

The multivector-plus-late-layer variant combines the two strongest signals observed so
far: a richer latent bank of goal vectors and late-layer goal availability. Use it to
test whether the late top-down-control benefit compounds with increased latent capacity,
or whether the two mechanisms mainly substitute for one another in this small pilot
regime.

The prefix variant is the first conditioning-mechanism diagnostic beyond additive
conditioning. It turns the latent goal state into a small bank of learned continuous
prefix tokens that are prepended to the generator context, testing whether persistent
goal access is better provided through a dedicated latent prefix than through additive
state injection.

The first follow-up after the base prefix pilot is a small prefix-token sweep. Compare
2, 4, and 8 latent prefix tokens under the same held-out Z-only strong-goal setting to
determine whether the competitive prefix result is robust or narrowly tuned to one
prefix length.

The prefix-KV condition keeps two goal-derived key/value memories available at every
self-attention layer. Unlike the continuous-prefix condition, these memories are never
queries and never occupy residual-stream output positions. A causal-mask regression
test verifies that adding this memory path does not expose future generator tokens.
Its checked-in pilot source is
[`results/z_only_prefix_kv_pilot_iteration1.json`](results/z_only_prefix_kv_pilot_iteration1.json).
At the matched one-seed scale it reaches .143 exact match, .609 facet satisfaction,
and a .049 shuffled-goal effect. This is below the matched two-token continuous prefix
and does not justify the additional layerwise K/V parameters in the shallow model.

The checked-in source for the matched direct-text comparison is
[`results/z_only_context_controls_pilot_iteration1.json`](results/z_only_context_controls_pilot_iteration1.json).
At this one-seed pilot scale the direct causal text prefix reaches .102 exact match and
.518 facet satisfaction, below both single-vector late-layer Z and the four-vector
continuous prefix. This remains an architecture diagnostic rather than a general claim:
the text condition processes a longer distractor-bearing prompt, whereas the latent
conditions separately encode the goal and generate from the content channel.

The eight-epoch text control is not a converged baseline. Its validation LM loss is
still 1.271, and the 40-epoch follow-up in
[`results/text_prefix_convergence_pilot_iteration1.json`](results/text_prefix_convergence_pilot_iteration1.json)
reaches .975 exact match and .995 facet satisfaction with validation LM loss .021.
Accordingly, the short-run comparison measures optimization speed and must not be used
to claim latent conditioning outperforms direct goal tokens.

The corresponding converged late-layer Z run is stored in
[`results/z_only_convergence_pilot_iteration1.json`](results/z_only_convergence_pilot_iteration1.json).
It improves to .197 exact match with a .102 strict shuffled-goal effect, but held-out
linear-probe facet accuracy remains .661 and isolated/full counterfactual success is
only .285. Gates 1--3 therefore do not permit the direct-access, depth, or horizon
sweeps yet; the next run must first isolate representation and channel capacity with a
canonical authoritative goal prompt.

The canonical-goal diagnostic removes templates, filler, and distractor blocks from the
encoder's goal-only input while retaining one authoritative requirement token per active
facet. Its reversed-order paraphrase tests order robustness. Generation still receives
the held-out content channel, and all prompt channels now contribute to the versioned
dataset SHA-256 rather than only the original full prompt.
Its checked-in result is
[`results/z_only_canonical_goal_pilot_iteration1.json`](results/z_only_canonical_goal_pilot_iteration1.json):
task exact match, held-out probe joint exact match, and complete isolated
counterfactual control all reach 1.0, while shuffled Z reduces exact match by .900.
This passes Gates 1--3 at one-seed diagnostic scale and permits a canonical direct-goal
exposure pilot. It does not establish a headline result or solve goal extraction from
distractor-bearing rendered prompts.

The direct-goal exposure curve keeps canonical late-layer Z active while inserting a
deterministically sampled fraction of authoritative goal blocks into the generator's
ordinary-token prefix. The completed canonical gate is the p=0 endpoint; four matched
configs add p in {.25, .5, .75, 1}. Compare correct and shuffled Z within every
condition to estimate whether the generator bypasses Z as a direct symbolic route is
restored.

The completed curve and plot source are
[`results/z_necessity_curve_pilot_iteration1.json`](results/z_necessity_curve_pilot_iteration1.json)
and [`results/z_necessity_curve_pilot_iteration1.csv`](results/z_necessity_curve_pilot_iteration1.csv).
Exact-match D_Z remains between .873 and .902 from p=0 to p=1 rather than declining
smoothly. At p=1, zero-Z facet satisfaction rises to .738, showing that direct tokens
are usable, but counterfactual Z still overrides conflicting direct requirements in
.926 of complete outputs. The current late-layer additive architecture therefore
prefers Z rather than bypassing it.

## Generation-horizon persistence

The horizon task expands content into a deterministic sequence of varying facet
requests and requires one goal-conditioned answer token for every scheduled position.
It uses all six facets and contains no target padding. The first point has five shuffled
six-facet schedules (31 output tokens including the end marker), comparing canonical
late-layer Z against a direct-only causal prompt. Evaluation reports satisfaction in
four output quartiles and early-minus-late drift for both conditions; shuffled-Z
counterparts are additionally reported only for the latent condition. Both conditions
instantiate a 1,152-position model so later horizon points remain parameter-matched.

The completed one-seed 31-token pilot is recorded in
[`results/horizon_32_pilot_iteration1.json`](results/horizon_32_pilot_iteration1.json),
with plot source in
[`results/horizon_32_pilot_iteration1.csv`](results/horizon_32_pilot_iteration1.csv).
Direct text reaches .637 per-position satisfaction versus .170 for correct Z and .090
for shuffled Z. Direct satisfaction falls from .764 in the first quartile to .483 in
the fourth, whereas the small .019 Z drift is not evidence of persistence because Z
is near floor throughout. Exact match is zero for both conditions, and both select the
last or penultimate epoch while validation loss is still improving. The optimization
gate therefore fails: converge this matched point before launching longer horizons.

All runs after the prefix-KV pilot use one shared exact-match definition for primary
and intervention outputs, including the required end token. Counterfactual evaluation
additionally reports full counterfactual exact match and isolated facet-swap success,
which requires the selected facet to change correctly while every untouched active
facet and the end token remain correct. Raw intervention generations are retained in
`predictions.jsonl` for auditability.

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
