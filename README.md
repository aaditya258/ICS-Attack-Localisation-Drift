# LENS-R

Code, settings and saved results for the paper

> A. Kumar, K. K. Singh, V. Kumar. *Comparative evaluation of static and adaptive reconstruction methods for attack localisation in industrial control systems under sensor drift.* Submitted to Computers and Electrical Engineering.

LENS-R ranks the sensors that may have caused an alarm in an industrial process, using only normal operating data. It removes slow sensor drift before ranking and reports the value that each flagged sensor should read. The paper compares it with static and recursive reconstruction-based contribution (RBC), with and without a five-minute exclusion gap, on the SWaT, HAI 22.04 and WADI testbeds.

## Repository layout

| Path | Contents |
|---|---|
| `src/recon.py` | LENS-R. Frozen before the WADI evaluation. Its SHA-256 is in `protocol/recon.sha256` |
| `src/adaptive_run.py`, `src/adaptive_faults.py` | Static and recursive RBC, injected sensor faults, drift near attack onset (post hoc) |
| `src/jpc_extra_tests.py` | Pre-registered claims, WADI exclusions, gap test and long-event horizon, from saved results |
| `src/jpc_figures.py` | Figures 1 to 4 of the manuscript |
| `src/*.py` (other) | Data loading, earlier designs, baselines, WADI run, sensitivity sweep, timing |
| `protocol/` | Pre-registration documents and the checksum of `recon.py` |
| `groundtruth/` | Attack tables with the evaluated target tags for SWaT A1, HAI 22.04 and WADI A2 |
| `reference_results/` | All outputs reported in the paper |
| `worked_example/` | One LENS-R update on SWaT attack 23, runs without the datasets |
| `data/` | Prepared datasets (not included, see `data/README.md`) |

## Pre-registration and post hoc analyses

Two plans were written before the corresponding results were computed. Both are in `protocol/`.

- `PREREG.md` (27 Sep 2026) fixed the evaluation of an earlier design on SWaT and HAI. That design did not meet its success criterion. LENS-R was developed afterwards on the same development attacks, so all SWaT and HAI results are development results.
- `PREREG_WADI.md` (27 Sep 2026, addendum 28 Sep 2026) fixed the single WADI evaluation of LENS-R before the WADI normal data were loaded. Claim A (accuracy) was met. Claim B (drift robustness) was not met, because LENS-R without drift correction lost 0.087 rather than more than 0.10.

Check that `recon.py` is unchanged:

```
sha256sum src/recon.py        # Linux
certutil -hashfile src\recon.py SHA256    # Windows
```

The comparison with static and recursive RBC, the injected sensor faults, the drift near attack onset and the five-minute horizon were added after the pre-registration and are post hoc.

## Quick checks without the datasets

```
pip install -r requirements.txt
python worked_example/worked_example.py      # reproduces the worked-example table
python src/jpc_extra_tests.py                # pre-registered claims and the extra tests, from reference_results
python src/jpc_figures.py                    # Figures 1 to 3 into ./figures
```

## Full reproduction

1. Obtain SWaT A1 and WADI A2 from iTrust. HAI 22.04 is downloaded automatically.
2. Prepare the data:
   ```
   python prepare_data.py --swat RAW_SWAT_DIR --wadi RAW_WADI_DIR
   ```
3. Run every step (about 1.5 to 2 hours on a two-core laptop):
   ```
   python run_all.py --list      # show the steps
   python run_all.py             # run all steps
   python run_all.py --only 16   # run a single step
   ```

Fresh outputs go to `./results`, `./cache` and `./figures`. Compare them with `./reference_results`.

## Where each result comes from

| Manuscript item | File in `reference_results/` |
|---|---|
| Accuracy on real attacks, LENS-R and RBC variants | `adaptive_dev_summary.csv`, `adaptive_dev_tests.csv`, `adaptive_wadi_summary.csv`, `adaptive_wadi_tests.csv` |
| Accuracy of the other baselines | `recon_swat_all.csv`, `recon_hai_all.csv`, `recon_pooled_all.csv`, `wadi_primary_summary.csv`, `wadi_secondary_keepconst_summary.csv` |
| Slow drift | `adaptive_hai_drift.csv`, `adaptive_wadi_drift.csv`, `inject_recon_summary.csv`, `wadi_inject_summary.csv` |
| Drift near attack onset | `adaptive_drift_speed_summary.csv` |
| Injected sensor faults | `adaptive_faults_summary.csv`, `adaptive_faults_per_injection.csv` |
| Five-minute horizon | `adaptive_horizon.csv`, `adaptive_horizon_per_attack.csv` |
| Ablation and sensitivity | `recon_pooled_all.csv`, `sweep_lensr.csv` |
| Supervised detectors and TreeSHAP | `summary_n10.json`, `recon_pooled_test.csv` |
| Computational cost | `adaptive_timing.csv` |
| Pre-registered claims, WADI exclusions, gap test | `wadi_claims.json`, `jpc_extra_tests.txt` |

## Acknowledgements

SWaT and WADI were provided by iTrust, Centre for Research in Cyber Security, Singapore University of Technology and Design. HAI was made publicly available by The Affiliated Institute of ETRI, Daejeon, South Korea.

## Licence

MIT. See `LICENSE`.
