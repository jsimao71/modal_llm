# AGENTS.md — Repository Handoff

This file is the continuity guide for future Codex sessions. Read
`docs/papers/AGENTS.md` before changing the research program; its scientific scope,
falsification rules, baselines, and stop condition remain authoritative.

## Current state

The active branch is `main`. The completed milestones are:

- `3c27e70` — Paper 0 was sharpened into a falsifiable position and research program.
- `b99b8c0` — Paper 1 received an executable B0--B10 experiment scaffold.
- `db5d018` — Paper 1 received held-out controls, staged training, pilot evidence,
  stronger reproducibility checks, and updated prose.

Both manuscripts use the shared bibliography at
`docs/papers/common/references.bib` through `\bibliography{../common/references}`.

Authoritative files:

- `docs/papers/AGENTS.md` — research instructions and scope.
- `docs/papers/paper0/paper0_position.tex` — Paper 0 manuscript.
- `docs/papers/paper1/paper1_encode_generate_review_validate.tex` — Paper 1 manuscript.
- `docs/papers/paper1/EXPERIMENTS.md` — experiment commands and artifact contract.
- `docs/papers/paper1/results/pilot_iteration2.json` — checked-in pilot source data.
- `src/modal_llm/` — experiment implementation.
- `configs/paper1/` — smoke, overfit, pilot, main, and baseline-suite configs.

## What has been implemented

Paper 1 implements the complete B0--B10 controlled ladder with identical instantiated
parameter counts and reported active parameter counts. The model supports explicit
encode/generate/review/validate modes, optional persistent `Z_goal`, review states,
validation, facet prediction, and causal interventions.

The current controlled generator is `independent-facets-v2`. It provides:

- independent binary requirements and separately stored facet labels;
- late authoritative requirements and delimited instruction-like distractors;
- counterfactual prompts/targets that flip one authoritative facet;
- train/validation/test prompt families `standard`, `reordered`, and `interleaved`;
- corruption families `single_flip`, `late_flip`, and mixed single/double flip,
  truncation, and invalid-end corruptions;
- disjoint deterministic namespaces by default and deliberate shared namespaces for
  memorization gates;
- SHA-256 hashes over fully materialized datasets, not only generator metadata.

Training and evaluation now provide:

- optional goal/facet warmup and same-goal paraphrase invariance;
- held-out validation checkpoint selection and bounded early stopping;
- one cached encode/goal computation per autoregressive candidate;
- separate training, selection-validation, base-generation, validator, and diagnostic
  compute counts;
- total/active parameters, approximate FLOPs, token positions, calls, and wall time;
- checkpoint reload verification;
- zero, shuffled, paraphrase, counterfactual-goal, and mode interventions;
- per-corruption validator metrics;
- immutable timestamped run directories and suite summaries;
- seed-paired baseline differences and normal-approximation 95% intervals.

The test suite covers generator determinism, split-family separation, corruption
coverage, counterfactual integrity, content hashes, parameter parity, active parameter
accounting, causal masks, bidirectional sensitivity, cached generation calls, seeded
optimizer determinism, finite losses, and all-nonfinite suite metrics.

## Current empirical result

These are seed-0, small-model, from-scratch implementation diagnostics. They are not
headline evidence.

| Protocol | Baseline | Exact match | Facet satisfaction | Zero-goal effect | Shuffled-goal effect | Validator AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Unstaged | B0 | 0.430 | 0.797 | — | — | — |
| Unstaged | B5 | 0.109 | 0.553 | 0.020 | 0.000 | — |
| Unstaged | B10 | 0.102 | 0.539 | -0.004 | 0.000 | 0.476 |
| Staged | B5 | 0.168 | 0.604 | 0.090 | 0.004 | — |
| Staged | B10 | 0.156 | 0.601 | 0.031 | -0.016 | 0.575 |

B0, B5, and B10 all reach 1.0 exact match in the shared-example memorization gate
(100, 250, and 250 optimizer updates). B10 also reaches 1.0 validator AUROC there.
The architecture and losses can therefore fit the data when generalization is removed.

The held-out result is negative: staged B5/B10 improve over their unstaged versions
but remain well below causal B0. Zeroing the staged goal has an effect, but shuffling
does not reliably hurt, so `Z_goal` is not yet demonstrably example-specific. The B10
validator improves to AUROC 0.575 and ranking accuracy 0.637 but is not ready for
selection or retry claims.

Do not run the expensive five-seed headline suite yet. Do not present these pilots as
confirmation of H1--H6.

## Recommended continuation

1. Strengthen example-specific goal learning. Test contrastive same-goal/different-goal
   objectives, more direct structured goal reconstruction, and conditioning mechanisms
   beyond the current additive projection, one change at a time.
2. Make shuffled and counterfactual goal substitutions training-independent acceptance
   gates. A useful goal state must degrade under mismatched substitution and alter the
   intended output facet under the one-facet counterfactual.
3. Strengthen the validator with balanced hard negatives and held-out corruption
   families. Require clear held-out calibration/ranking improvement before using it for
   best-of-N, retry, or revision.
4. Repeat the small held-out pilot after each isolated change. Preserve B0 and
   compute/parameter controls. Predeclare acceptance thresholds rather than selecting
   them after seeing results.
5. Only after the causal gates pass, run `configs/paper1/baselines.yaml` across five
   seeds and report paired effects with uncertainty.
6. Then extend beyond the synthetic vocabulary to deterministic transformations,
   executable/code tasks, natural instruction templates, and pretrained backbones.
7. Add mechanistic probes only when the behavioral effect is stable enough to explain.

PRA, hierarchical goals, tools, RL, and online consolidation remain future integration
work under `docs/papers/AGENTS.md`. No PRA code has been copied into this repository.
If shared utilities are later needed, inspect `D:/git/rd/pdattention/src/common` and copy
only the smallest justified pieces under `src/modal_llm/common`, retaining provenance
and tests.

## Reproduction commands

From the repository root:

```powershell
python -m pip install -e .
pytest
python -m compileall -q src tests
modal-llm train-eval --config configs/paper1/smoke.yaml
modal-llm suite --config configs/paper1/smoke-suite.yaml
modal-llm train-eval --config configs/paper1/overfit.yaml
modal-llm suite --config configs/paper1/overfit-suite.yaml
modal-llm suite --config configs/paper1/pilot-suite.yaml
modal-llm suite --config configs/paper1/pilot-staged-suite.yaml
```

Build Paper 1 from `docs/papers/paper1`:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error paper1_encode_generate_review_validate.tex
```

The last verified state had 12 passing tests, successful Python compilation, a complete
B0--B10 smoke suite with a timestamped aggregate, successful checkpoint reloads, and a
nine-page Paper 1 PDF with resolved bibliography and cross-references. PyTorch may emit
a harmless nested-tensor warning because `norm_first=True`; MiKTeX may emit locale and
bibliography underfull-box warnings.

## Working rules

- Run `git status --short` before editing and preserve unrelated user changes.
- Never stage raw `results/`, checkpoints, LaTeX build products, or unrelated files.
- Check in only curated table/plot source data under the paper directory.
- Keep raw run directories immutable; never reuse or overwrite them.
- Update the manuscript when implementation details or evidence change.
- Run tests, compile Python, build the affected paper, and run `git diff --check` before
  committing.
- Commit and push after each meaningful paper update so another machine can continue.

At the time this handoff was written, the local worktree also contained unrelated
uncommitted `.gitignore` changes, generated PDFs, and raw `results/`. They were
deliberately excluded from the Paper 1 commits and must not be accidentally staged.
