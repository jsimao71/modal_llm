# AGENTS.md --- Codex Instructions for Multi-Mode Transformer Cognition

## Mission

Build a rigorous, reproducible program for explicit cognitive modes and
persistent goal/task state. First target:

`ENCODE -> GENERATE -> REVIEW -> VALIDATE`

Do **not** prematurely implement hierarchical planning, PRA integration,
general tool execution, or uncontrolled continual learning. Keep future
integration interfaces clean.

## Primary question

Can explicit functional modes plus persistent latent goal state improve
task fidelity and self-evaluation relative to parameter- and
compute-controlled decoder-only transformers?

## Hypotheses

-   **H1:** bidirectional task encoding improves globally constrained
    task representation.
-   **H2:** latched `Z_goal` improves constraint retention during
    generation.
-   **H3:** bidirectional review of completed output improves evaluation
    representation.
-   **H4:** comparing intended `Z_goal` with realized `Z_out` improves
    validation/ranking/retry.
-   **H5:** explicit mode identity enables differentiated computation
    with shared weights.
-   **H6:** any benefit must survive parameter/data/inference-compute
    controls.

## Falsification rules

Explicitly test whether encode is merely extra compute; Z is redundant
with prompt context; review is redundant with another causal pass;
validator learns shortcuts; mode embeddings add nothing beyond masks;
retry gains are only extra sampling; self-validation fails to correlate
with external correctness. Remove failed components rather than
protecting them.

## V0 architecture

### Mode

`mode in {encode, generate, review, validate}` with learned mode
embeddings. Mask is not the only mode signal.

### Encode

Input original task; bidirectional mask; produce `H_goal`.

Goal extraction occurs **before vocabulary projection**. Compare mean
pooling, special task token, learned goal query, then 4--8 learned facet
queries. Start with one vector.

### Goal state

`Z_goal = GoalHead(H_goal)`. Initially latch it for the whole
generation.

Conditioning variants, one at a time: 1. additive/broadcast residual
projection; 2. prefix-like latent KV; 3. lightweight cross-attention; 4.
later, goal-conditioned residual gates.

### Generate

Ordinary causal decoding conditioned on Z.

### Review

After complete candidate Y: `H_out = F_theta(Y, review, bidirectional)`
`Z_out = ReviewHead(H_out)`

Ablate Y-only review vs `(task,Y)`, causal second pass, and final
generation residual only.

### Validate

`score, facets, DeltaZ = Validator(Z_goal, Z_out)`

Use external/deterministic labels whenever possible.

## Inference policies

Implement separately: no validation; validation only; best-of-N;
threshold retry; discrepancy-conditioned revision. Always log total
tokens, forward passes, approximate FLOPs, and wall time.

## Dataset program

Create controlled generators first: - K independent constraints with
varied position/length; - critical late constraints; - instruction-like
distractors; - ordered multi-part requirements; - deterministic
transformations/arithmetic/symbolic tasks; - small executable/code
tasks; - correct/incorrect candidate pairs differing in one facet.

Store ground-truth facets separately.

## Required baselines

B0 decoder-only; B1 parameter-matched decoder; B2 compute-matched
decoder; B3 bidirectional encode/no Z; B4 causal encode+Z; B5
bidirectional encode+Z; B6 generation-final-state review; B7
bidirectional review; B8 validator rereads prompt+answer/no Z; B9
validator uses Z+output representation.

## Losses

Keep configurable:
`L = L_LM + lambda_goal L_goal + lambda_val L_val + lambda_rank L_rank + lambda_aux L_aux`

Goal objectives may include facet decoding, paraphrase
invariance/contrastive identity, structured requirement reconstruction,
predictive usefulness.

Validation: BCE/multilabel BCE, calibrated regression, pairwise ranking,
externally verified reward prediction. Do not start RL until validator
calibration is established.

## Metrics

**Task:** accuracy, EM, F1, pass rate, per-facet satisfaction,
long-output retention.

**Validator:** AUROC/AUPRC, Brier, ECE, ranking accuracy, oracle gap,
confidently-wrong false positives.

**Representation:** Z stability, same-goal paraphrase similarity,
different-goal separation, probes, CKA/SVCCA where useful, causal
interventions.

**Mechanistic:** residual update norm/alignment, refinement vs
complementarity/interference, attention entropy/dilution, head
similarity, temporal stability, mode specialization.

## Causal interventions

Zero Z; shuffle Z; substitute same-goal paraphrase Z; replace one facet;
perturb learned facet direction; remove mode embedding but retain mask;
swap mode embeddings; freeze goal head; fixed vs recomputed Z.

## Roadmap

A shared weights + mode/masks\
B single-vector Z\
C multi-facet Z\
D review/output facets\
E validator + DeltaZ\
F retry/revision\
G goal-conditioned residual gates\
H multiple goals/transitions\
I progressive tools/callbacks\
J RL over cognitive transitions\
K controlled sleep/consolidation

## Integration contracts

**PRA:** future typed references may include
document/task/tool/skill/observation. PRA owns selective
materialization.

**Gated residuals:** reserve `gate(h_l, Z, mode)`; not mandatory in
Paper 1.

**Tools:** future action `(tool_id, params, goal_id)` and callback
`(event_id, goal_id, status, observation)`. Security/runtime remain
external.

**Goal hierarchy:** later root/active/parent/pending/blocked/completed
states, dependencies, suspension/resumption, completion evidence. Do not
keep all goals continuously active.

## Sleep/consolidation

No uncontrolled online base-weight updates. Use immutable base,
provenance-aware replay, adapter-only candidates, held-out regression
tests, promotion gate and rollback. Treat user feedback and harness
telemetry as noisy evidence.

## Reproducibility

Every run records git commit, config, seed, dataset hash, parameter
count, optimizer, hardware, wall time, tokens/forward passes, metrics,
and plot/table source artifacts. Use multiple seeds for headline claims.
Never overwrite raw results.

## Suggested repository structure

-   `src/mmtc/` model/modes/goal/review/validator
-   `datasets/` generators/schemas
-   `experiments/` configs
-   `eval/` external validators/metrics
-   `analysis/` mechanistic probes
-   `tests/` masks/parity/shapes/causality/determinism
-   `results/` immutable run summaries
-   `papers/` manuscripts

## Coding rules

Prefer minimal transformer modifications. Unit-test masks. Preserve
baseline parity when features are disabled. Separate
model/eval/orchestration. Keep CPU tiny smoke tests. Do not optimize
CUDA before architecture stabilizes.

## Paper-1 stop condition

Paper 1 ends after rigorously answering: Does bidirectional encode help?
Does persistent Z help? Does review help? Does intended-vs-realized
comparison improve validation? Does validation improve inference under
compute control? What do mechanistic probes reveal?

Long-term principle: test whether functions currently forced into
homogeneous autoregressive computation benefit from explicit
representations, dynamics, and selection mechanisms.
