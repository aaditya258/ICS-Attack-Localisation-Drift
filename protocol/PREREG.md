# LENS v2 — pre-registered design and analysis plan

Written 2026-09-27, before any LENS v2 or new-baseline result was computed.
Earlier exploratory runs (LENS-U, simple drift fix, LENS-DA, LENS-SW) used the same test attacks;
this document fixes the confirmatory analysis so that LENS v2 is evaluated exactly once.

## Method (all parts label-free; everything learned from normal data only)
Notation: z = tag standardised with normal-data mean/sd (as in LENS); updates every 3 s.

1. Context-conditioned residual. For every tag i, a ridge regression (lambda = 1.0 on standardised
   features, fitted on normal data sampled every 5 s) predicts z_i(t) from
   (a) all other tags at t, and (b) the tag's own level L_i(t) = mean of z_i over [t-600 s, t-300 s).
   Rationale for (b): the lag must be much shorter than drift timescales (hours) and longer than
   attack onset / fast process dynamics. Residual e_i = z_i - prediction. Discrete tags (actuators,
   setpoints) use the same model, so their score is the surprise of the state given the context.
   sigma0_i = robust (MAD) scale of e_i on normal data, floored at 0.05.
2. Online new-normal adaptation. Per tag, bias b_i and scale s_i are EWMAs of e_i and 1.2533 |e_i - b_i|
   with time constant 1800 s (alpha = 3/1800), initialised at 0 and sigma0_i. Score
   u_i = min(|e_i - b_i| / max(s_i, 0.5 sigma0_i), 50). Updates of tag i are frozen while u_i > 3,
   for at most 30 min in a row (then resume). Runs from the start of each stream.
3. Onset precedence. Per-tag CUSUM: S_i = min(max(0, S_i + u_i - 2), 500). Accumulation favours tags that
   deviate first and persistently over later, downstream tags.
LENS v2 score at an update = S_i, normalised to sum to 1 (ranking unaffected by normalisation).

## Ablations (reported, not selected among)
- no own-history term (1 without (b)); no adaptation (u from sigma0 only); no CUSUM (u_i directly);
- static context residual without own history, instantaneous (closest to earlier label-free relevance).

## New baselines
- Conditional reference, CondAttr-style simplification: for tag i, the K = 20 nearest normal states
  (30,000 normal samples, standardised raw space) using all tags except i; score = |z_i - neighbour mean|
  / scale_i (scale from 2,000 held-out normal samples). The original uses a learned latent space.
- Linear forecast residual, GDN-style: ridge (lambda = 1.0) predicting z(t) from the mean of all tags over
  [t-10 s, t-1 s]; score = |e - median| / IQR of normal residuals.
Existing: TreeSHAP on RF, raw deviation, simple drift fix, LENS-U, LENS-SW, RF Gini, random.

## Evaluation protocol (unchanged)
Localisable attacks only; score averaged over the first 10 updates (30 s) after labelled onset;
metrics Top@1/3/5, stage hit, MRR (ties averaged); bootstrap 95% CI over attacks; Wilcoxon signed-rank
on per-attack reciprocal rank.

## Primary comparisons
Pooled test set (SWaT 11 + HAI 24 = 35 attacks), LENS v2 vs:
(P1) simple drift fix, (P2) conditional reference (CondAttr-style), (P3) TreeSHAP on RF.
Secondary: all 91 localisable attacks (label-free methods only); per-dataset results; ablations.
Success criterion for the claim "LENS v2 improves on existing label-free references":
higher pooled-test MRR than P1 and P2 with p < 0.05 in at least one, and not significantly worse than P3.

## Drift-injection experiment (HAI, all 58 attacks)
Slow level shift added to standardised units: offset(t) = m * sigma_i * min(1, (t - t_start)/7200 s),
i.e. a 2-hour ramp from the start of each test file, then held (recalibration-style shift).
m in {0, 3, 6, 10}.
- Scenario D (distractors): 5 continuous tags that are never an attack target, chosen with seed 0.
- Scenario T (targets): all attack-target tags of each file (pre-attack drift of the attacked tags).
Models are NOT refitted on drifted data. Report MRR vs m for LENS v2, its ablations, simple fix, LENS-U,
LENS-SW, raw deviation, conditional reference and linear forecast residual.
Expected if the mechanism works: LENS v2 stays flat as m grows; fixed-reference methods degrade in D.
