# AGENTS.md â€” Modal LLM Paper 1, Next Iteration

## Mission

Run the next Paper 1 iteration as a mechanistic diagnostic of persistent goal state, not as a headline benchmark.

The current pilot showed:
- staged B5/B10 improve but remain below the causal baseline;
- zeroing `Z_goal` can hurt;
- shuffling `Z_goal` barely hurts;
- therefore `Z_goal` is used, but is not yet reliably example-specific.

The next iteration must determine whether:
1. `Z_goal` fails to encode the goal;
2. `Z_goal` encodes it but generation bypasses it;
3. `Z_goal` matters mainly when intent must persist across long generation;
4. `Z_goal` matters mainly in deeper networks;
5. the conditioning mechanism is too weak;
6. shared causal/bidirectional weights interfere.

Do not run expensive multi-seed headline experiments until `Z_goal` has a clear causal role.

---

# Core hypothesis

The refined hypothesis is:

> Persistent goal state should become more valuable as the required goal influence must survive longer computational trajectories in token time and/or network depth.

Define:

`Delta_Z(D,T) = Score_Z(D,T) - Score_baseline(D,T)`

where:
- `D` = transformer depth;
- `T` = generation horizon.

Test whether:

`d Delta_Z / dT > 0`

and/or:

`d Delta_Z / dD > 0`.

A valid result may be that Z is neutral or harmful on short/shallow tasks but beneficial on long/deep ones.

---

# Part A â€” Make Z the only goal-information channel

## A1. Split task input

Represent each task as:

`X = (C, G)`

where:
- `C` = content/data;
- `G` = goal/task specification and constraints.

Encode only G:

`Z_goal = Encode(G)`

Generation receives:

`Y = Generate(C, Z_goal)`

and MUST NOT receive the original G tokens.

This creates a hard necessity test: the requested behavior must be communicated through Z.

## A2. Required controls

Run:

- **A0 Full direct baseline:** generation receives `(C,G)` causally, no Z.
- **A1 Z-only:** generation receives `C + correct Z`.
- **A2 Shuffled-Z:** generation receives `C + Z` from another example.
- **A3 Zero-Z:** generation receives `C + 0`.
- **A4 Constant-Z:** same fixed/learned Z for every example.
- **A5 Random-Z:** random Z matched approximately in scale.

Primary gate:

`Score(correct Z) >> Score(shuffled Z)`

If this does not hold on simple held-out synthetic tasks, stop and fix representation/conditioning before larger sweeps.

---

# Part B â€” Context/prefix controls

The goal is to determine whether any benefit comes from persistent goal state specifically or simply from giving the generator a compressed prefix representation.

Compare:

- **B0 Text prefix:** normal goal tokens before content.
- **B1 Continuous prefix:** goal encoder produces K continuous prefix vectors.
- **B2 Prefix-KV:** goal encoder produces latent prefix K/V.
- **B3 Single-vector Z:** current residual/additive conditioning.
- **B4 Multi-vector Z:** `Z in R^(K x d)`, initially K in `{4,8}`.
- **B5 Repeated Z:** inject the same Z at multiple/all layers.

Keep each variant separately selectable and ablatable.

---

# Part C â€” Goal decodability versus goal use

Answer two separate questions:

1. Does Z contain the goal?
2. Does generation use the goal in Z?

Train simple held-out probes on frozen `Z_goal` for machine-readable facets such as:
- operation type;
- output format;
- count;
- inclusion/exclusion constraints;
- ordering;
- target entity/class;
- constraint ID.

Report:
- linear-probe accuracy/F1;
- small-MLP probe accuracy/F1;
- joint facet exact match.

Interpretation:

- High probe + weak shuffle effect => encoding works; conditioning/use fails.
- Low probe + weak shuffle effect => representation itself fails.
- High probe + strong shuffle effect => Z is an example-specific causal goal channel.
- Good train but weak held-out => overfitting to task/prompt family.

---

# Part D â€” Counterfactual interventions

Construct matched examples with identical C but different G.

Examples:
- ascending vs descending;
- JSON vs CSV;
- include A vs exclude A;
- 3 bullets vs 5 bullets;
- sum vs maximum.

Run:

`Generate(C, Z_A) -> Y_A`

`Generate(C, Z_B) -> Y_B`

Behavior must change in the predicted direction.

For multi-facet Z, add facet swaps where feasible:
`[z1_A, z2_A, z3_A] -> [z1_A, z2_B, z3_A]`

and test whether only the corresponding output requirement changes.

This is stronger evidence than probes alone.

---

# Part E â€” Restore direct X progressively: Z-necessity curve

After Z works as the only goal channel, gradually restore direct access to goal information.

Let:

`p_direct in {0.0, 0.25, 0.5, 0.75, 1.0}`

control how much of G generation can access directly.

Implement through one or more controlled methods:
- masking subsets of goal tokens;
- revealing only selected goal facets;
- probabilistic direct-goal exposure;
- explicit direct-vs-latent channel switches.

Define:

`D_Z(p) = Score(correct Z,p) - Score(shuffled Z,p)`

Research question:

> Does the model bypass persistent Z as a shorter/direct token route becomes available?

Expected possible signature:

`D_Z(0) >> 0`

with decreasing `D_Z(p)` as `p -> 1`.

If observed, treat this as an architectural finding.

---

# Part F â€” Generation-horizon sweep

Test whether persistent Z becomes more useful as generated output becomes longer.

Use target horizons approximately:

`T in {32,64,128,256,512}`

Optionally add `1024` if tasks remain meaningful and compute allows.

Do not pad meaningless text merely to increase T.

## Long-horizon tasks

Build tasks where the same goal must affect the whole output:
- repeated global formatting rule;
- forbidden class/token/category;
- required entity/field in every segment;
- ordering rule maintained throughout;
- multi-section output controlled by one original goal;
- long structured transformation;
- delayed requirement that affects a decision far into generation.

## Position-aware metrics

Measure constraint satisfaction over output position, e.g.:
- 0â€“25%;
- 25â€“50%;
- 50â€“75%;
- 75â€“100%.

Define:

`Drift(T) = Satisfaction_early - Satisfaction_late`

Compare baseline vs Z.

The relevant signature may be slower late-output degradation rather than higher short-output accuracy.

---

# Part G â€” Network-depth sweep

Test depth approximately:

`D in {4,8,16,24,32}`

adjusted for available compute.

Keep width/tokenizer/data/optimization as comparable as possible. Report parameter counts.

Where feasible include:
1. fixed-width depth sweep;
2. approximate parameter-matched depth/width control.

## Z injection variants

Compare:

- **Z0 input-only:** inject before first layer only.
- **Z1 early:** first quarter of layers.
- **Z2 periodic:** every K layers.
- **Z3 all-layer:** inject every transformer block.
- **Z4 late:** final quarter only.

This is central to the persistent top-down-control hypothesis.

If repeated access to Z is the important mechanism, Z3 should gain relative to Z0 as depth grows.

---

# Part H â€” Layerwise goal persistence

Measure how goal influence evolves through depth.

For each layer compare correct-Z and shuffled-Z runs:

`I_l = distance(h_l(correct Z), h_l(shuffled Z))`

Use:
- cosine distance;
- normalized L2;
- CKA over batches where useful;
- residual-update differences;
- output-logit divergence;
- layerwise goal probe accuracy.

Ask:
- Does goal information decay?
- Does it remain stable?
- Does repeated Z injection restore it?
- Does it become progressively integrated?

---

# Part I â€” Joint depth x time sweep

Once Parts Aâ€“H pass, run the central grid.

Example:

- `D in {4,8,16,32}`
- `T in {32,128,512,1024}`

For each cell compare:
1. causal baseline;
2. correct-Z model;
3. shuffled-Z intervention;
4. prefix baseline where practical.

Compute:

`Delta_Z(D,T) = Score_Z(D,T) - Score_B0(D,T)`

and:

`Causal_Z(D,T) = Score_correctZ(D,T) - Score_shuffleZ(D,T)`

Report simple trends versus:
- `log(T)`;
- D;
- interaction `D x log(T)`.

Do not overfit complex statistics to pilot-scale data.

---

# Part J â€” Distinguish temporal and transformational persistence

Keep two mechanisms conceptually separate.

## Temporal persistence
Goal must survive many autoregressive steps.

Primary variable: `T`.

## Transformational persistence
Goal must survive many residual transformations.

Primary variable: `D`.

The interaction may be important, but do not conflate the mechanisms.

---

# Part K â€” Shared-weight interference diagnostic

If useful, measure gradient compatibility between modes:

`cos(grad L_encode, grad L_generate)`

and later:

`cos(grad L_review, grad L_generate)`.

If strongly negative or unstable, add a diagnostic condition with tiny mode-specific adapters/LoRA:

`F_(theta, phi_encode)`
`F_(theta, phi_generate)`

Theta remains shared.

This is diagnostic only; do not make mode-specific networks the default without evidence.

---

# Part L â€” Validator workstream

Continue validator work separately.

Compare:
- **V0:** validator reads original text goal + output;
- **V1:** validator reads `Z_goal + Z_out`;
- **V2:** validator reads `Z_goal + output tokens`;
- **V3:** full prompt + output with equivalent compute.

Report:
- AUROC;
- AUPRC;
- Brier score;
- ECE;
- ranking accuracy;
- confidently-wrong false positive rate;
- best-of-N oracle gap.

Interpretation:
- strong text validator + weak Z validator => latent representation problem;
- weak both => task/corruption/training problem;
- strong Z validator => intended-vs-realized latent comparison is viable even before generation improves.

---

# Part M â€” Training progression

Use staged training unless an ablation explicitly disables it.

## Stage 1 â€” Goal representation
Train encoder/goal head on explicit facets and/or contrastive goal identity.

## Stage 2 â€” Z-only generation
Train generation where G is unavailable except through Z.

## Stage 3 â€” Causal gate
Require strong held-out correct-vs-shuffled effect.

## Stage 4 â€” Restore direct goal route
Run the Z-necessity curve.

## Stage 5 â€” Depth and horizon sweeps
Only after Z is demonstrably causal.

## Stage 6 â€” Review/validation
Strengthen validator after representation quality is understood.

Do not begin RL in this iteration.

---

# Part N â€” Go/no-go gates

Before multi-seed headline experiments require:

### Gate 1 â€” Goal decodability
Held-out facets strongly decodable from Z.

For simple synthetic facets, target roughly >90% where task difficulty permits.

### Gate 2 â€” Causal identity
In Z-only tasks:

`Score(correct Z) - Score(shuffled Z)`

is large and stable.

### Gate 3 â€” Counterfactual control
Changing one goal facet while keeping C fixed produces the predicted behavioral change.

### Gate 4 â€” Persistence signature
At least one sweep shows increasing Z value with T and/or D, or lower late-output drift.

### Gate 5 â€” Reproducibility
Checkpoint reload reproduces predictions; tests pass; configs/seeds/results are complete.

If Gates 1â€“3 fail, do not run the full depth x time grid.

---

# Part O â€” Required plots

Produce at least:

1. correct vs shuffled Z on Z-only tasks;
2. Z-necessity curve vs `p_direct`;
3. constraint satisfaction vs generation position;
4. score vs generation horizon T;
5. score vs depth D;
6. heatmap of `Delta_Z(D,T)`;
7. heatmap of `Causal_Z(D,T)`;
8. layerwise goal probe/sensitivity;
9. validator calibration/ranking if validator experiments run.

Preserve CSV/JSON source data for every plot.

---

# Part P â€” Required run metadata

Each run should record:

```yaml
model:
  depth:
  hidden_size:
  heads:
  parameters:
  z_type:
  z_vectors:
  z_injection_layers:
  mode_adapters:

task:
  family:
  content_length:
  goal_facets:
  target_output_length:
  direct_goal_exposure:

training:
  seed:
  stages:
  train_tokens:
  optimizer:
  learning_rate:
  early_stopping:
  checkpoint_metric:

evaluation:
  exact_match:
  facet_satisfaction:
  early_satisfaction:
  late_satisfaction:
  goal_drift:
  zero_z_effect:
  shuffled_z_effect:
  random_z_effect:
  goal_probe:
  validator_metrics:
  tokens_generated:
  forward_passes:
  wall_time:
```

Preserve raw outputs.

---

# Part Q â€” Paper update requirements

Update Paper 1 with a section tentatively titled:

## Persistent goal state as a causal channel

Document the current pilot honestly:
- Z has nonzero influence;
- shuffle produces little example-specific effect;
- therefore presence/use of Z is insufficient evidence of goal representation.

Use the stronger operational criterion:

> A latent state represents the goal to the extent that interventions on that state, while holding relevant non-goal content fixed, cause predictable goal-consistent changes in behavior.

Add the depth/horizon hypothesis:

> Persistent goal state may provide little benefit for short/shallow computation but become increasingly valuable when intent must remain influential across many autoregressive steps or residual transformations.

Do not imply positive results before experiments are run.

---

# Part R â€” Interpretation guide

### Outcome A
Z works when direct goal tokens are removed, but loses influence when they return.

Interpretation: viable latent channel plus bypass preference for direct token routes.

### Outcome B
Z causal effect grows with generation horizon.

Interpretation: support for temporal-persistence hypothesis.

### Outcome C
Z causal effect grows with depth, especially with repeated injection.

Interpretation: support for transformational-persistence / top-down-control hypothesis.

### Outcome D
Z remains weak even in Z-only setting.

Interpretation: representation/conditioning architecture still fails; stop and fix.

### Outcome E
Goal probes are strong but behavior ignores Z.

Interpretation: conditioning mechanism is the primary failure.

### Outcome F
Repeated per-layer Z succeeds while input-only Z does not.

Interpretation: persistence/top-down availability matters more than one-shot compression.

### Outcome G
Continuous prefix matches or beats Z.

Interpretation: compact goal representation is useful, but the current special Z mechanism is not yet justified.

### Outcome H
Small mode adapters substantially improve results.

Interpretation: shared-mask/shared-weight mode interference may be real.

All outcomes are valid scientific results.

---

# Final principle

This iteration is not trying to prove that explicit goal state improves LLMs.

It is trying to determine **when, where, and under what causal conditions a persistent latent goal variable becomes functionally meaningful**.

Execution order:

`force Z to matter`
`-> prove Z contains the goal`
`-> prove behavior depends on the correct Z`
`-> restore direct context`
`-> sweep generation horizon`
`-> sweep model depth`
`-> measure persistence`
`-> only then run multi-seed headline experiments`
