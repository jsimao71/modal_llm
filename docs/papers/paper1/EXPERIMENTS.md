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
modal-llm train-eval --config configs/paper1/horizon-32-z-converged.yaml
modal-llm train-eval --config configs/paper1/horizon-32-direct-converged.yaml
modal-llm train-eval --config configs/paper1/horizon-32-z-prefix-6.yaml
modal-llm train-eval --config configs/paper1/horizon-32-z-multivector-prefix-6.yaml
modal-llm train-eval --config configs/paper1/horizon-32-z-facet-slot-prefix.yaml
modal-llm train-eval --config configs/paper1/horizon-32-z-facet-slot-signaled.yaml
modal-llm train-eval --config configs/paper1/horizon-32-z-facet-slot-signaled-converged.yaml
modal-llm train-eval --config configs/paper1/horizon-64-direct.yaml
modal-llm train-eval --config configs/paper1/horizon-64-direct-converged.yaml
modal-llm train-eval --config configs/paper1/horizon-64-z-facet-slot-signaled.yaml
modal-llm train-eval --config configs/paper1/horizon-64-z-facet-slot-signaled-converged.yaml
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
The convergence follow-up changes only the maximum joint budget from 40 to 80 epochs
and early-stopping patience from 5 to 8 validations. Data, initialization seed,
architecture, optimizer, and evaluator remain fixed.

The extended result is stored in
[`results/horizon_32_extended_optimization_iteration1.json`](results/horizon_32_extended_optimization_iteration1.json),
with plot source in
[`results/horizon_32_extended_optimization_iteration1.csv`](results/horizon_32_extended_optimization_iteration1.csv).
Direct text reaches 1.0 exact match and 1.0 satisfaction in all quartiles, proving that
the scheduled task is learnable and that its pilot drift was an optimization artifact.
Correct Z reaches only .176 satisfaction and zero exact match, barely above its
40-epoch .170 result; shuffled-Z satisfaction is .090, for a causal effect of .086.
The single additive vector is therefore used but fails to support accurate
query-dependent retrieval over the varying schedule. Longer horizons remain gated;
the next comparison should test structured latent access at the same 31-token point.
The first structured-access control keeps the single-vector encoder fixed and projects
that Z into six continuous prefix tokens. Generated positions can attend to these
latent memory tokens at every layer, isolating addressable presentation from increased
encoder-state capacity. It uses the same 80-epoch ceiling and all other horizon data
and optimizer settings.

That presentation control is complete in
[`results/horizon_32_structured_prefix_iteration1.json`](results/horizon_32_structured_prefix_iteration1.json).
It reaches .172 satisfaction and zero exact match, versus .176 and zero for matched
late-layer additive Z. Its correct-minus-shuffled satisfaction effect is .093, slightly
larger than additive Z's .086, but the small .007 quartile drift again occurs near
floor. Six addressable tokens projected from one Z therefore do not repair retrieval,
despite using more active parameters. The next isolation should increase encoded-state
structure rather than only expanding a single vector after encoding.

The next control holds the six-token continuous-prefix interface fixed while changing
the encoder from one pooled goal vector to six learned-query goal vectors. This tests
encoded-state structure separately from prefix length. The learned vectors are not
assumed to be facet-aligned; facet-specific claims require subsequent swaps and probes.
All task, optimization, seed, and evaluation settings remain matched.

The multi-vector result is stored in
[`results/horizon_32_multivector_prefix_iteration1.json`](results/horizon_32_multivector_prefix_iteration1.json),
with the matched conditioning comparison in
[`results/horizon_32_conditioning_comparison_iteration1.csv`](results/horizon_32_conditioning_comparison_iteration1.csv).
Six vectors reach .185 satisfaction and zero exact match, only .013 above the
one-vector prefix condition despite 1.61 times its active parameters. Validation goal
loss falls sharply from .0726 to .00329, but the shuffled-Z effect remains .093. This
decodability-use gap localizes the main failure to query-specific generator access,
not raw latent capacity. Do not add more generic vectors or increase horizon next;
test explicit facet-to-memory coupling or aligned supervision.

The active-parameter ratio above corrects an instrumentation error discovered during
the next implementation step: earlier runs counted every available conditioning
projection as active even when the selected mode did not execute it, and omitted
direct learned queries. Total parameter counts and all behavioral metrics were
unaffected. Corrected active counts are 205,830 for one-vector prefix and 331,014 for
six-vector prefix; the immutable raw run artifacts retain their originally reported
counts and total-parameter-based approximate FLOPs for provenance. Future approximate
FLOPs use corrected executed-mode active counts.

The aligned-slot diagnostic adds two opt-in mechanisms. `goal_pooling: facet_tokens`
extracts each canonical requirement into its known facet slot and applies a shared
slot-level value objective. `conditioning_mode: slot_prefix` exposes those slots
directly as six latent memory tokens without flattening or remixing them. Evaluation
also replaces one slot with its matched counterfactual and reports selected-facet
satisfaction, untouched-facet satisfaction, and full counterfactual exact match.

The aligned-slot result is stored in
[`results/horizon_32_facet_slot_prefix_iteration1.json`](results/horizon_32_facet_slot_prefix_iteration1.json).
It reaches .161 satisfaction and zero exact match, below generic six-vector prefix.
Correct-minus-shuffled satisfaction is .081. More decisively, replacing the selected
facet slot with its counterfactual changes the corresponding scheduled positions
correctly only .008 of the time; full counterfactual exact match is zero. Explicit
slot extraction, slot-level value supervision, and direct slot presentation therefore
do not establish localized behavioral control. Longer horizons and facet-swap claims
remain gated. Next diagnose latent-prefix token signaling or encode--generate gradient
compatibility rather than adding capacity.

The next interface control preserves the aligned slots and direct slot prefix but adds
two signals carried by ordinary generation tokens: a learned facet-slot identity and
the slot's generation-prefix position embedding. The failed unsignaled condition
remains selectable and reproducible. A gain would localize the deficit to latent-token
signaling; another failure would strengthen the case for an optimization-compatibility
diagnostic.

The signaled-slot result is stored in
[`results/horizon_32_facet_slot_signaled_iteration1.json`](results/horizon_32_facet_slot_signaled_iteration1.json).
Adding slot identity and generation position raises correct-Z satisfaction from .161
to .743 and the shuffled-Z effect from .081 to .369. A counterfactual slot swap now
achieves .513 satisfaction on selected-facet positions and .738 on untouched positions,
large gains from .008 and .187. However, task and counterfactual exact match remain
zero, epoch 80 is selected, and validation LM is still declining at .308. This is the
first strong long-schedule Z mechanism result, but it is not converged or complete.
Extend this exact condition at 31 tokens before increasing horizon.
The convergence extension changes only the joint ceiling from 80 to 120 epochs and
patience from 8 to 10 validations. It preserves the seed, initialization trajectory,
data, model, optimizer, slot interface, and causal evaluator.

The convergence result is stored in
[`results/horizon_32_facet_slot_signaled_convergence_iteration1.json`](results/horizon_32_facet_slot_signaled_convergence_iteration1.json).
Correct Z reaches 1.0 exact match and satisfaction in every quartile. Shuffled Z falls
to .0156 exact match and .503 satisfaction, yielding causal effects of .984 and .497.
Replacing one facet slot produces the complete counterfactual sequence in .672 of
examples while preserving untouched positions at .998 satisfaction. The aligned,
signaled interface therefore passes the one-seed 31-token task and causal gates after
sufficient optimization. Localized control remains incomplete and epoch 120 is still
selected, so advance only to the next approximately 64-token diagnostic point rather
than the full grid or headline replication.

The next horizon point uses ten independently shuffled six-facet blocks, yielding 61
output tokens including the end marker. Direct text and signaled facet-slot Z retain
the same seed, two-layer 64-wide architecture, data sizes, optimizer, 120-epoch ceiling,
and causal evaluator used at the converged 31-token point. Evaluation batch size falls
from 16 to 8 only for memory safety. Run direct text first as the task-learnability
gate, then correct/shuffled/counterfactual Z.

The 120-epoch direct-text gate is under-optimized rather than passed. It obtains zero
exact match and .678 facet satisfaction, with quartile satisfaction
(.809, .490, .673, .739). Epoch 120 is selected and validation LM is still falling
rapidly (1.100 at epoch 100 to .473 at epoch 120). Consequently, do not run or
interpret the matched Z condition yet. Extend only the direct-text optimization ceiling
with seed, data, architecture, optimizer, and evaluator fixed. Exact values and the gate
decision are in `results/horizon_64_direct_iteration1.json`.

The direct convergence extension changes only the joint-training ceiling from 120 to
160 epochs and keeps patience at ten validations. It starts from the same deterministic
seed rather than reloading the epoch-120 checkpoint because checkpoints do not preserve
AdamW optimizer state; resetting optimizer moments would introduce an unplanned change.

The extension passes the direct 61-token learnability gate: exact match, aggregate
satisfaction, and all four temporal quartiles are 1.0, with zero drift. Validation LM
falls to .00194 at the selected epoch 160. The trajectory exactly reproduces the prior
run through epoch 120 at reported precision (.473413), showing that the earlier failure
was delayed optimization rather than demonstrated horizon capacity. The matched
signaled facet-slot Z condition may now run with the same 160-epoch ceiling. Exact
values are in `results/horizon_64_direct_convergence_iteration1.json`.

The matched Z run uses the same 160-epoch ceiling as the passed direct gate. Relative
to the staged 120-epoch Z configuration, only that ceiling changes. Correct-Z task
behavior remains the first gate; shuffled-Z necessity and one-slot counterfactual
locality are interpreted only if correct Z learns the 61-token schedule.

The 61-token Z condition fails the task and locality gates despite the matched direct
pass. Correct-Z exact match is zero and satisfaction is .160 in nearly flat quartiles
(.167, .157, .156, .160), versus shuffled-Z satisfaction .079. The .081 graded effect
shows residual example-specific use, but selected and untouched slot-swap satisfaction
are both approximately .163, so control is not localized. This is not the direct run's
under-optimization pattern: the selected epoch is 151 of 160 and validation LM has
plateaued near 1.12, while validation goal loss reaches .00194. The resulting
decodability--use gap blocks the remaining horizon grid. Next isolate repeated slot use
and generation-time injection at the learned 31/61-token boundary. Exact values are in
`results/horizon_64_z_facet_slot_signaled_convergence_iteration1.json`; the compact
comparison is in `results/horizon_persistence_iteration1.csv`.

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
