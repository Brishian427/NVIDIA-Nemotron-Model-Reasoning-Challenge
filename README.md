# Post-Training a 30B Reasoning Model: A Diagnostic Post-Mortem

LoRA fine-tuning of Nemotron-3-Nano-30B-A3B for the NVIDIA Nemotron Model
Reasoning Challenge (Kaggle).

At archive time, the safe submitted candidate showed an unofficial leaderboard
score of about **0.86** and rank **142**. The final official certificate had not
yet been issued when this repository was prepared.

This repository is less about the final score and more about the diagnostic
reasoning behind post-training: how optimization signals mislead, how models
forget, and how to allocate scarce compute when every training run costs hours
and every submission is one of a handful.

---

## TL;DR

The competition asked participants to fine-tune a 30B MoE reasoning model to
solve nine categories of "Alice's Wonderland" logic puzzles. A strong
open-sourced winning adapter scored around 0.85 out of the box.

What follows is honest about outcomes:

- **The result that shipped (about 0.86)** came from a small, well-understood
  increment on top of the winning adapter, not a from-scratch method.
- **The ambitious branch failed.** An attempt to teach the model a hard puzzle
  category through distilled chain-of-thought collapsed the model twice
  (0.86 to 0.33, then to 0.40). The root cause was never confirmed, because the
  tooling needed to inspect raw model outputs never ran on the target GPU.
- **The most valuable output is the diagnostic methodology** developed along the
  way, which transfers directly to other post-training settings, including RL.

If you are here to judge whether I can reason about training dynamics, start
with [`docs/methodology/`](docs/methodology/) and
[`docs/postmortem/the-0.40-mystery.md`](docs/postmortem/the-0.40-mystery.md).

---

## Why this might interest someone working on post-training / RL

This was a supervised fine-tuning (SFT) project, not RL. But several of the
problems it surfaced are the same problems that dominate RL post-training, just
in a different guise:

| Problem encountered here (SFT) | Its RL counterpart |
|---|---|
| Healthy training loss, collapsed eval score | Reward goes up, policy gets worse |
| Mean loss hides a minority category collapsing | Aggregate reward hides mode collapse or narrow exploitation |
| Training loss measures fitting, not capability | Train reward measures the proxy, not the objective |
| Held-out accuracy as the only trustworthy signal | Held-out eval vs. train reward separation |
| Catastrophic forgetting under narrow data | Forgetting under continual or online updates |
| Scarce GPU and submission budget | Sample-efficient experimentation under expensive rollouts |

The recurring lesson is simple: **a healthy optimization curve is not evidence
that the model got better**. Here it showed up in SFT, and it cost two
submissions before it was understood.

---

## The three training runs

### Run 1: narrow fine-tuning caused catastrophic collapse (0.33)

The first ambitious branch fine-tuned purely on 320 new chain-of-thought traces
for one hard category. Training loss descended cleanly to 0.13. The submission
scored **0.33**, far below the 0.86 baseline.

This was not ordinary forgetting. It was a whole-model collapse: the model had
been over-specialized on a single category and lost broad task competence.

Diagnosis: **catastrophic forgetting**, driven by the absence of replay data
plus overfitting on a tiny corpus.

### Run 2: anti-forgetting mix had a healthy log but collapsed result (0.40)

The intervention was to mix the 320 new traces with 1,500 proportionally sampled
original examples across all nine categories, train for a single epoch from the
winning adapter, and halve the learning rate.

The training log was the cleanest of all three:

- mix data was confirmed loaded after a prior silent fallback bug was fixed
- one epoch, no visible overfitting
- loss descended smoothly with healthy batch-to-batch variance
- gradients were stable, with no explosions

**The submission still scored 0.40.**

This is the crux of the project. Every signal visible during training said
"healthy." The result said "broken." The only signals that could have caught
this - raw model outputs or per-class held-out accuracy - were exactly the ones
we did not have.

### Why the collapse was never confirmed

Two independent paths to inspect the model's actual outputs hit the same wall:
the inference tooling failed to initialize on the competition's Blackwell GPU
due to a compiler permission error before generation could start.

So the honest status of the ambitious branch is: **not falsified, just never
verified**. The algorithmic work underneath it is real and measured; what failed
was the last mile of distilling it into the model, for a reason we could not
observe.

That distinction - "unverified" is not "disproven" - is itself a useful
discipline.

---

## Methodology

These are written up as standalone documents because they outlive this
competition.

### 1. Loss-curve health diagnosis

See [`docs/methodology/loss-curve-diagnosis.md`](docs/methodology/loss-curve-diagnosis.md).

The absolute value of loss is meaningless by itself; its normalized position is
what carries information.

- **Starting point / ln(vocab_size)** tells you whether the model already knows
  the task. A ratio near 0 means you may be re-training on things the model
  already knows.
- **End / start ratio** measures how much was learned and warns of overfitting.
- **Batch-to-batch variance** fingerprints data heterogeneity: smooth descent
  often means homogeneous data; healthy jitter can indicate a real mix.
- **Gradient clipping** is a safety net, not a performance knob. If it triggers
  constantly, the real problem is usually the learning rate.

### 2. The examiner dashboard

See [`docs/methodology/examiner-dashboard.md`](docs/methodology/examiner-dashboard.md).

Mean loss over mixed data is diluted: 1,500 easy examples at tiny loss can drown
out 320 hard ones at meaningful loss, so the mean tells you almost nothing about
the category you actually care about. Two layers fix this:

- **Per-class training loss**: is this category actually being learned, or just
  riding along?
- **Held-out accuracy**: did the model learn the method, or memorize the answer
  style?

The second layer is the one this project lacked, and its absence is why two
submissions were spent blind.

### 3. Cost-tiered experiments

See [`docs/methodology/cost-tiered-experiments.md`](docs/methodology/cost-tiered-experiments.md).

Arrange experiments by ascending cost, and let each cheap layer gate the
expensive one:

```text
free local data check -> cheap proxy experiment -> expensive training run -> submission
        veto                     veto                      veto
```

A cheap veto earns its keep by eliminating bad options before expensive
resources are committed. Its value is in rejection, not approval.

---

## The algorithmic work

The hardest category was a cryptarithm-style task where the winner's deterministic
reasoner solved only about **8%** of the training examples. By reframing it as
**joint constraint synthesis** - treating all examples in a puzzle as constraints
on one global symbol-to-digit mapping - solver accuracy rose from **8% to 59%**
on the train split.

That number is a solver result: it measures an algorithm solving the puzzles.
Turning it into a model capability through distillation is the branch that failed
above. Keeping those two separate - an algorithm that works vs. a model that
learned it - is the honest framing.

---

## Engineering note: the silent data bug

One run reported "healthy" while training on the wrong dataset entirely: a
notebook silently fell back to the original 7,830-example corpus instead of the
intended 1,820-example mix. Nothing errored. The log looked fine. Only the
starting-loss ratio, near zero, gave it away in hindsight: the model already knew
almost all of the data.

The fix was to make a missing dataset a loud failure, not a silent substitution:
hardcode the path, assert the exact example count, and crash immediately if
either is wrong. A failed startup is far cheaper than hours spent training on the
wrong data.

See [`docs/postmortem/silent-data-bug.md`](docs/postmortem/silent-data-bug.md).

---

## Repository layout

```text
docs/methodology/   transferable diagnostic methods
docs/postmortem/    the 0.40 mystery and the silent data bug
docs/full-wiki.md   sanitized long-form working log
src/                cryptarithm synthesis, CoT generation, data mixing, diagnostics
notebooks/          sanitized training notebooks
results/            training logs and figures
```

---

## Credit

The 0.86 working result builds directly on the open-sourced winning adapter for
this competition. This project's contribution is the incremental training on top
of it, the diagnostic methodology, and the unfinished algorithmic exploration,
not the base solution. Standing on that shoulder is gratefully acknowledged.

---

## License

MIT. See [`LICENSE`](LICENSE).
