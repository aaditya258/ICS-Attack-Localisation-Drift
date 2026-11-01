# LENS-R — pre-registered confirmatory evaluation on WADI

Written 2026-09-27, after development on SWaT and HAI and before any WADI file was opened.
The WADI run is executed once; no setting below may change after WADI data is loaded.

## Frozen method (recon.py, as committed with this file)
LENS-R T2 (full): standardisation and consistency model M = regularised normal precision matrix (ridge 0.01),
fitted on WADI normal data only. Slow stage: mean over [t-1800 s, t-300 s), greedy repair of up to 10 tags while
J exceeds the 99.9th percentile of slow-averaged normal J; offsets subtracted (drift correction).
Fast stage: mean over (t-10 s, t], greedy repair K = 5, scores = consistency gains. Updates every 3 s.

## Data handling (fixed in advance)
WADI.A2 (2019) 14-day normal file and labelled attack file. Columns with more than 50% missing or constant in
normal data are dropped; remaining gaps forward-filled then back-filled. Ground truth = attack points from the
official attack description, mapped to column names; attacks whose points cannot be mapped are excluded and
listed. Localisation window = first 10 updates (30 s) after labelled onset; metrics as before
(Top@1/3/5, stage hit, MRR with ties averaged; bootstrap CI; Wilcoxon signed-rank).

## Comparisons (all label-free, all localisable WADI attacks)
(P1) GDN-style linear forecast residual; (P2) LENS-R without drift correction; (P3) simple drift fix;
(P4) CondAttr-style kNN; plus raw deviation and random for reference.

## Pre-registered claims
A. Accuracy: LENS-R MRR is not significantly lower than any of P1-P4 (no comparison with p < 0.05 in their favour),
   and is higher than P3.
B. Drift robustness (HAI protocol applied to WADI: 5 non-target continuous tags, seed 0, 2-hour ramp to m sigma,
   m in {0, 6, 10}, models not refitted): LENS-R MRR falls by less than 0.05 from m = 0 to m = 10, while
   P1 and P2 each fall by more than 0.10.
Both A and B are reported whatever the outcome.

## Addendum (2026-09-28, written after inspecting labels and attack table, before loading normal data or running any method)
- Timestamps: the Time column of WADI_attackdataLABLE.csv keeps only mm:ss. Rebuilt as 2017-10-09 18:00:00 + row index (1 Hz);
  verified: minutes/seconds and dates agree on 100% of 172,801 rows; 13 labelled episodes start 3 s after the table times.
- Label error: attack 13 (table 2017-10-11 12:07:30-12:10:52) is labelled at 00:07:33-00:10:55 (12-hour shift). Its onset is
  taken as table time + 3 s (12:07:33), consistent with the other attacks. The mislabelled 00:07 episode is ignored for
  localisation. Results are also reported excluding attack 13.
- Attack 14 (chemical dosing stop) is excluded: the description does not name the dosing tag. Attack 10 is scored on
  1_MV_001_STATUS only (the damaged raw-water pump is not identified). Attack 8 (reported unsuccessful) is kept; results
  are also reported without it.
- Localisable WADI attacks: 13 (1, 2, 3-4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15).
