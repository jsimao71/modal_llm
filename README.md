# Multi-Mode Transformer Cognition (MMTC)

## Core hypothesis

Decoder-only transformers force task encoding, generation, review,
validation, and eventually planning/action through one homogeneous
causal next-token process. This research program tests whether a shared
transformer substrate with explicit **cognitive mode** and persistent
latent **goal/task state Z** improves goal fidelity, long-horizon
stability, self-evaluation, and later reinforcement learning/control.

The first papers deliberately do **not** attempt full planning, tool
use, or AGI. The strategy is:

`isolate -> measure -> understand -> falsify/refine -> cross-pollinate -> integrate`

## Parallel research lines

-   **PRA / Progressive Retrieval Attention:** selective memory and
    bounded materialization.
-   **Rich/gated residuals:** selective computation and
    refinement/complementarity/interference.
-   **Multi-mode + Z:** goal/state control, review, validation, and
    cognitive phases.
-   **Progressive tooling/controller:** persistent tool representations,
    action selection, callbacks, execution state.

Long-term integration hypothesis:

`goal selection -> memory selection -> computational selection -> action selection -> observation -> goal update`

## Paper roadmap

### Paper 0 --- Position

`paper0_position.tex`: **Beyond Homogeneous Next-Token Prediction: A
Research Program for Mode-Structured, Goal-State Transformers**

Defines the motivation, architecture space, falsifiable research
program, links to the other lines, and a multi-paper roadmap.

### Paper 1 --- Minimal architecture

`paper1_encode_generate_review_validate.tex`: **Encode, Generate,
Review, Validate: Explicit Cognitive Modes with Persistent Goal State in
a Shared Transformer**

Tests:

`ENCODE -> Z_goal -> GENERATE -> REVIEW -> Z_out -> VALIDATE`

with shared transformer weights where possible, bidirectional
encode/review masks, causal generation, explicit mode embeddings, a
latched goal representation, and validation from intended vs realized
state.

### Candidate Paper 2 --- Goal-conditioned computation

Mode- and goal-conditioned residual gating.

### Candidate Paper 3 --- Goal dynamics

Multiple goals; active/pending/blocked/completed state; completion
inference; hierarchy; suspension/resumption; learned subgoal generation.

### Candidate Paper 4 --- Native action/tool control

Persistent tool representations, progressive tool retrieval, structured
action channels, asynchronous callbacks, and model-level task state.

### Candidate Paper 5 --- Consolidation

Experience replay, feedback-derived data, safe adapter-only sleep-time
learning, regression gates, and RL over cognitive transitions.

## Paper-1 minimal architecture

Let `m in {ENCODE, GENERATE, REVIEW, VALIDATE}` and:

`H^(m) = F_theta(X, Z, e_mode(m), M_m)`

1.  **Encode:** bidirectionally encode the complete request.
2.  **Goal extraction:** `Z_goal = Q_goal(H_encode)` before vocabulary
    projection.
3.  **Generate:** causal decoding conditioned on latched `Z_goal`.
4.  **Review:** bidirectionally re-encode completed output.
5.  **Output extraction:** `Z_out = Q_review(H_review)`.
6.  **Validate:** `V(Z_goal, Z_out) -> score, facet scores, DeltaZ`.
7.  Optional best-of-N, retry, or discrepancy-conditioned revision.

## Experimental principles

-   Begin with **same weights, different mode**.
-   Add small mode-specific adapters only after the shared-weight
    baseline is understood.
-   Use parameter-, data-, and compute-matched baselines.
-   Externally verify correctness; never equate self-score with success.
-   Separate goal fidelity from generic answer quality.
-   Test `Z` against simply re-reading the prompt.
-   Test bidirectional encode independently of `Z`.
-   Test review against equivalent extra causal compute.
-   Preserve negative results.

## Initial task families

Controlled multi-constraint instruction following; late constraints;
distractors; ordered multi-part outputs; long-generation requirement
retention; deterministic transformations; arithmetic/symbolic tasks;
small executable/code tasks; candidate pairs differing in one
requirement.

Store machine-readable requirement facets separately from prompt text.

## Metrics

**Outcome:** accuracy, EM/F1, pass rate, constraint satisfaction, task
completion, executable tests.

**Goal state:** Z stability; paraphrase invariance; different-goal
separation; facet probes; causal sensitivity to Z interventions.

**Validation:** AUROC/AUPRC, Brier/ECE, best-of-N ranking,
false-positive rate, error localization, improvement per extra
FLOP/token.

**Mechanistic:** attention dilution; residual update magnitude;
refinement/complementarity/interference; mode-conditioned representation
similarity; layer/head specialization; temporal feature stability; goal
influence on residual updates.

## Training progression

1.  Standard LM/pretrained base.
2.  Mixed causal/bidirectional mode training.
3.  Goal/facet objectives.
4.  Review/validation supervision.
5.  Ranking/preference learning.
6.  Controlled retry/revision.
7.  RL from externally grounded outcomes.
8.  Later: goals, tools, callbacks, hierarchical RL.

## Sleep/consolidation principle

Do not initially modify deployed base weights continuously. Accumulate
provenance-aware experience, train candidate adapters (e.g. LoRA)
offline/idle-time, run regression gates, and promote only validated
adapters. Likes, edits, retries, tests, tool outcomes, and controller
choices are possible evidence, not automatically trustworthy labels.

## Files

-   `docs/papers/AGENTS.md`: research and reproducibility contract.
-   `docs/papers/paper0/paper0_position.tex`: position paper.
-   `docs/papers/paper1/paper1_encode_generate_review_validate.tex`: Paper 1 protocol.
-   `docs/papers/common/references.bib`: shared bibliography.
-   `src/modal_llm/`: controlled generator, shared-mode model, metrics, and runner.
-   `configs/paper1/`: smoke, main, and B0--B10 suite configurations.
-   `tests/`: masks, parity, determinism, shapes, and loss smoke tests.

## Run Paper 1 experiments

```powershell
python -m pip install -e .
pytest
modal-llm train-eval --config configs/paper1/smoke.yaml
modal-llm suite --config configs/paper1/smoke-suite.yaml
```

Use `configs/paper1/main.yaml` for the main B10 run and
`configs/paper1/baselines.yaml` for the five-seed B0--B10 comparison. Results
are written to new timestamped directories and include configs, provenance,
dataset hashes, checkpoints, raw predictions, calibration/task metrics, causal
interventions, compute counts, and approximate FLOPs.

The `.bib` file is a seed bibliography and should be verified against
authoritative metadata before submission.
