# Adapter-MoE — design draft (NOT scheduled; post-v9 arm candidate)

> **Status:** design sketch, written 2026-09-09 at the user's request.
> Predecessors required: v7's result (in flight on the second PC) and
> **v8a's news-domain data must land first** — otherwise every expert
> inherits the same Wikipedia-domain gap and routing is pointless.
> Nothing in this file overrides the v7/v8/v9 queue.

## 1. The idea

Stop averaging every skill into one adapter. Train **one LoRA per skill
bucket** and route each benchmark item to its expert with a deterministic
prompt-shape router. The project's own history justifies the
decomposition: every cycle that added a slice *dedicated* to a skill beat
the field on that skill (arithmetic 6.8→26.9, sequence emission 3→82% with
accuracy above zero-shot, Counterfactual +13pp from 502 rows) — while
single adapters paid trade-offs between buckets (§ 4b of the handoff).

Positioning vs prior art: multi-adapter composition exists (AdapterFusion,
LoRAHub-style merging) — typically merged, not routed, and typically
justified post-hoc. This arm's difference: **the expert decomposition is
derived from a measured bucket-leverage analysis (D42), and the router is
deterministic and auditable.**

## 2. Experts (4) — each a normal LoRA cycle on the existing machinery

| # | Expert | Mixture (draft) | Target bucket (TIME share) |
|---|---|---|---|
| E1 | MCQ | v7-style: `AUG_MCQ2`-heavy + context-grounded rehearsal | MCQ answerable 44.7% (0.45pp/pt — D42) |
| E2 | arithmetic | `AUG_ARITH` + TimeQA date-math core | Computation (9,372) + TimeBench arithmetic |
| E3 | ordering | `AUG_SEQ` + `AUG_DURATION` + `AUG_RELATIVE` | sequence 10.8% + ordering/duration cats |
| E4 | free-text QA (**fallback**) | core + long-form dolly (v6 recipe) | free text 40.4%; default route |

Each expert's eval needs only its own bucket → eval cost per expert is a
fraction of a full TIME pass. Total: 4 × ~12 h training + partial evals
≈ **2 GPU-days**.

## 3. Router — deterministic prompt-shape rules, zero learned parameters

- choices present in the prompt → E1
- self-specified letter-sequence format (`_SELF_SPECIFIED_FORMAT_RE`, the
  rule the standard prompt already uses) → E3
- arithmetic cue words ("how long", "years between", date spans) → E2
- otherwise → E4
- **Legality constraint:** the router may read only prompt-visible
  features — never gold answers, never benchmark category labels. Routing
  must be reportable as a confusion matrix against gold-shape.

A learned router (tiny classifier) is an optional ablation AFTER the
rule-based one works; it is not the primary arm.

## 4. Protocol and comparison

- **Comparison target:** the single best adapter at the time (v7 or v8a)
  on identical data pools, plus zero-shot. If experts train on different
  data than the single adapter, reviewers will attribute any win to the
  data change, not the routing — hold the pools equal in aggregate.
- **Metrics:** overall TIME/TimeBench; per-bucket accuracy (free, by
  construction); routing confusion matrix; expert forgetting outside its
  bucket (spot-check each expert on 200 off-bucket items).
- **Pre-registered risks:** router errors cost score directly at the
  misrouted bucket's rate; no-answer items need abstention behavior in
  E1 (v7's shortcut-free `AUG_NOANS` recipe); attribution is clean per
  bucket — this arm answers the D39 "combined arms don't attribute"
  critique by design.

## 5. Prerequisites (hard)

1. v8a domain data validated — all experts must train on news-domain data.
2. The best single adapter (v7/v8a/v9 winner) exists as the baseline.
3. Mistral transfer decision resolved (if mistral gets the winning recipe
   first, run adapter-MoE on one model only — llama — to bound scope).
4. A `mistral` data_loader port (implementation review 2026-09-09, risk 1)
   if any expert trains mistral.

## 6. What would make it a thesis win

- Beats the best single adapter overall AND on ≥2 buckets, with the
  confusion matrix showing routing accuracy ≥ ~97% (shape rules should be
  near-perfect; if they are not, that is itself a finding).
- Gives the thesis a second methodological contribution beyond the data
  fixes: **specialize-and-route**, with the bucket-leverage analysis as
  its design justification.
