# Exploratory source class-conditional conformal comparison

| Type | Date | Status |
| --- | --- | --- |
| report | 2026-10-24 | current |

24 October 2026. Fixed classifier predictions; no new training or target-label fitting. The selected model and serving thresholds remain unchanged. This post hoc comparator addresses the observed class-mix/coverage failure and is separate from the registered SC-only H2 analysis.

## Rule and validation

At alpha=0.1, compute a finite-sample LAC threshold separately from the true-label scores of each source calibration class. Use that threshold for the corresponding candidate label in every target prediction set. Target labels never enter calibration. If a class is absent or its finite-sample quantile falls beyond the available observations, its threshold is infinite and the label is always included. This is a conservative missing-support policy, not an inferred probability. The pooled comparator uses the same source calibration rows and a single LAC quantile.

[Fact] Evaluated PAFA and the fixed three-BEATs mean, with no SC and trained static SC, on five seeds and four LODO devices: 160 group/arm/seed/fold/rule rows. Every source and target probability array passes the existing annotation-order/alignment and finite-probability checks. An assert-based self-check verifies per-label quantiles, absent-label inclusion and threshold broadcasting. Source/input SHA-256 hashes are recorded.

Uncertainty: 1,000 paired hierarchical draws per group/arm/device. Resample model seeds, then whole source calibration and target patients; recalibrate every class threshold and the pooled threshold in each draw. Intervals are pointwise exploratory intervals, not corrected for testing multiple models/devices/rules. They do not account for training-patient resampling or new devices.

## All-device results

| Group | SC | Device | Pooled coverage | Class-conditional coverage [95% interval] | Pooled size | Class-conditional size |
| --- | --- | --- | ---: | --- | ---: | ---: |
| PAFA | no_SC | Meditron | 0.918 | 0.922 [0.898, 0.939] | 1.942 | 2.523 |
| PAFA | no_SC | LittC2SE | 0.857 | 0.890 [0.839, 0.918] | 1.951 | 2.446 |
| PAFA | no_SC | Litt3200 | 0.966 | 0.980 [0.922, 0.989] | 2.761 | 3.221 |
| PAFA | no_SC | AKGC417L | 0.809 | 0.833 [0.764, 0.877] | 2.262 | 2.446 |
| PAFA | static | Meditron | 0.902 | 0.903 [0.870, 0.921] | 1.771 | 2.295 |
| PAFA | static | LittC2SE | 0.880 | 0.907 [0.862, 0.932] | 1.859 | 2.168 |
| PAFA | static | Litt3200 | 0.975 | 0.980 [0.923, 0.994] | 3.007 | 3.298 |
| PAFA | static | AKGC417L | 0.700 | 0.863 [0.788, 0.901] | 1.890 | 2.389 |
| U_beats3 | no_SC | Meditron | 0.953 | 0.934 [0.899, 0.951] | 2.352 | 2.710 |
| U_beats3 | no_SC | LittC2SE | 0.938 | 0.944 [0.901, 0.960] | 2.199 | 2.559 |
| U_beats3 | no_SC | Litt3200 | 0.986 | 0.943 [0.828, 0.987] | 3.000 | 3.285 |
| U_beats3 | no_SC | AKGC417L | 0.821 | 0.861 [0.797, 0.891] | 2.170 | 2.540 |
| U_beats3 | static | Meditron | 0.948 | 0.928 [0.884, 0.943] | 2.356 | 2.763 |
| U_beats3 | static | LittC2SE | 0.954 | 0.960 [0.922, 0.973] | 2.173 | 2.453 |
| U_beats3 | static | Litt3200 | 0.992 | 0.953 [0.849, 0.989] | 3.272 | 3.473 |
| U_beats3 | static | AKGC417L | 0.753 | 0.857 [0.789, 0.900] | 1.957 | 2.393 |

All per-seed class counts, class coverage, singleton/empty rates and always-included labels are retained in the JSON. Paired coverage/set-size contrasts and prior-matched absolute-error changes are also retained; coverage intervals must not be reused as uncertainty for set size or classification.

## Selected PAFA/static on AKGC417L

![Source calibration has a higher normal-class share than held-out AKGC417L. Class-conditional thresholds raise crackle coverage but wheeze and combined crackle/wheeze coverage remain below 90 percent.](../../outputs/evaluation/report-20261024/class-conditional/akgc417l-class-coverage.png)

Bars are means over the five seed-specific model/calibration outcomes; open points show every seed. Target class mix is fixed and shown once. Source calibration mix varies with the seed-specific source-patient split. Both axes start at zero; all values and points are retained. This is a selected diagnostic of the previously identified difficult-coverage fold; the all-device table above prevents interpreting it as representative of every fold. Standalone [SVG](../../outputs/evaluation/report-20261024/class-conditional/akgc417l-class-coverage.svg) and [PDF](../../outputs/evaluation/report-20261024/class-conditional/akgc417l-class-coverage.pdf) exports and exact data are available. Publisher-specific requirements have not been specified.

| True class | Target cycles | Pooled coverage | Class-conditional coverage |
| --- | ---: | ---: | ---: |
| Normal | 1912 | 0.996 | 0.913 |
| Crackle | 1515 | 0.367 | 0.876 |
| Wheeze | 485 | 0.791 | 0.776 |
| Both | 304 | 0.360 | 0.618 |

[Fact] Pooled coverage rises from 0.700 to 0.863; gain +0.162 [+0.066, +0.246]. Mean set size grows from 1.890 to 2.389; increase +0.499 [+0.041, +1.244]. These are paired, pointwise exploratory estimates, not a confirmatory success claim.

[Fact] One seed has only eight combined crackle/wheeze calibration cycles; its finite-sample threshold for that class is infinite, so the class is always included. Its 100% inclusion point is displayed, not removed. Sparse calibration support can therefore increase coverage by increasing uncertainty rather than by improving classification.

[Interpretation] The pooled threshold gives nearly complete normal coverage but low crackle/combined-class coverage. Reweighting calibration by conditioning on class repairs much of the crackle shortfall, consistent with class mixture contributing to the pooled failure. Residual wheeze/combined-class undercoverage remains; correcting class mix alone is insufficient. These observations cannot attribute the residual to hardware alone because patient, site and device differ together.

No classifier Sp/Se/Score changes in this experiment. Class-conditional calibration does not guarantee coverage under arbitrary device shift or nonexchangeable cycles within patients. The same source-class score distribution must transfer for the usual conditional-coverage argument, and this evaluation does not establish that assumption. A macro above nominal does not establish every-device reliability. Larger sets and lower singleton rates are explicit costs. Do not replace registered H2, claim SOTA, or change serving thresholds from this comparison.

## Reproduction

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH=. .venv-train/bin/python \
  outputs/evaluation/report-20261024/class-conditional/analyse.py
MPLCONFIGDIR=/tmp/auscult-mpl-cache OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH=. \
  .venv/bin/python outputs/evaluation/report-20261024/class-conditional/figure.py
```

Archive `summary.json`, `figure-data.json`, `analyse.py`, `figure.py`, `audit.log` and PNG/SVG/PDF under `outputs/evaluation/report-20261024/class-conditional/`. The figure was rendered and visually checked; its legend was moved outside the data area, all five seed points remain visible, and exact plotting values/provenance are saved.
