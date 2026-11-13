# EHB paper plots

This plotting-only repository consumes immutable reduced EHB outputs. It does
not retrieve ERA5, calculate heat budgets, or estimate climatologies. Use the
shared `dev_env`. This repository's `environment.yml` records the plotting
dependency contract, consistent with the calculation's scientific libraries
and including the existing Seaborn style dependency. Exact execution versions
are recorded in each temperature-context manifest. Do not rebuild a shared
environment merely to run these plots.

## Results with temperature context

```bash
python scripts/make_plots.py --dataset DATASET --data-root INPUT_ROOT \
  --start 2021-06-01T00:00:00 --end 2021-08-31T23:00:00 \
  --plots-root NEW_OUTPUT_ROOT --no-diagnostic-plots \
  --stage1-file STAGE1.nc --temperature-climatology-file CLIMATOLOGY.nc \
  --tas-threshold-file THRESHOLD.nc
```

Supply all three auxiliary files together. Without them, the top panel contains
only volume-averaged temperature. Volume is never drawn, but remains essential
to budget normalization. Existing budget aggregation and diagnostic figures are
unchanged. The three results retain their filenames (hourly, 24-hour smoothed,
and daily). Existing nonempty figure directories are not overwritten.

- `STAGE1.nc`: contract-v2 HW_analysis harmonized regional timeseries. Use
  `tas_region`, a Kelvin-valued regional **daily mean** repeated by calendar day
  on the hourly analysis axis. Do not interpret it as hourly T2m.
- `CLIMATOLOGY.nc`: HW_analysis regional hourly climatology, using `T_mean`
  (source `T_domain_avg`). Match month, day, and UTC hour, not day-of-year.
  The initial PNW surface-to-700 hPa product uses the fixed 1940-2024 arithmetic
  mean, without event exclusion. Its recorded Stage-1 checksum must match.
- `THRESHOLD.nc`: ERA5 TAS evolving q90 output, with `climatology` and `threshold`
  indexed by year and a 365-day calendar with February 29 omitted. Select the
  plotted year, subtract the leap-day offset after February, and reject February
  29 itself. These existing products use 31-year and 31-day windows; no new
  mean, percentile, interpolation, or extrapolation is computed here.

The production `run_info.json` supplies `cli_args.region`, `request.bbox`,
`request.zg_top_pressure`, and `request.zg_bottom` for domain validation.
Auxiliary regions and vertical bounds must agree. Coverage, finite values,
calendar coordinates, daily T2m constancy, and temperature units are checked
before any figure is written. Temperature context currently requires contiguous
whole-hour budget samples and ISO date or whole-hour datetime selection bounds.

Some legacy EHB `T_domain_avg` arrays carry the volume-mask attributes
`units="1"` and `long_name="Fractional layer volume above surface pressure"`.
EHB `terms.compute_T_domain_average` defines this field as `integral(T dV)/V`,
in Kelvin. The loader accepts this exact inherited-attribute case only if every
selected value agrees with Kelvin-labelled Stage-1 `T_mean` (explicitly sourced
from `T_domain_avg`) within eight float64 epsilons times the local magnitude.
This resolution is recorded in the manifest and printed in the run log; inputs
are not modified. Unknown or unverified units are rejected. Threshold units
`same as input` resolve against the validated Kelvin TAS input and are recorded.

Observed temperatures are solid, climatological means dashed, and the blue
band spans the evolving T2m mean to q90. There is no percentile band for the
volume mean. Daily T2m and both daily references remain steps, including in the
24-hour-smoothed figure. Only the hourly volume mean and its hourly climatology
receive the same centered rolling mean (`min_periods=1`); the daily figure
averages their selected hourly samples. A figure note identifies both distinct
baseline conventions. Result figures use a 6-by-10-inch single-column canvas
with 18 pt labels, ticks and titles, and 14 pt legends and baseline notes.
This follows the HW_analysis single-column typography convention; reducing
the figure to a 3.5-inch manuscript column gives approximately 10.5/8.2 pt
text. Top-panel height remains 1.6 times either lower panel. Legends and the
baseline note wrap within the canvas, and export preserves the exact canvas
at 300 DPI (1800 by 3000 pixels).

The result palette matches the corresponding variables in the HW_analysis
PNW top-event figures at commit `3736891`: volume temperature `#E60000`,
temperature tendency and diabatic heating `#D55E00`, advection `#2636CA`,
and adiabatic heating `#0DAF43`. T2m and its references remain `#0072B2`.
The single-column font context is restricted to result rendering; diagnostic
figures retain their existing dimensions and 14/9 pt theme. Data loading,
daily sums, hourly smoothing, uncertainty bands and temperature references
are unchanged. Existing datasets are sufficient to regenerate the figures.

Temperature-context runs write `plot_manifest.json` beside the figure folders,
including source checksums, unit resolutions, baseline definitions, the EHB
and plotting identities, environment, selected timestamps, layout, and output
checksums. Regenerate into a fresh output root for every publication revision.

## Validation and Venus

```bash
python -c 'import sys; print(sys.executable)'
MPLCONFIGDIR=/tmp/ehb-paper-mpl python -W error -m pytest -q
python scripts/make_plots.py --help
bash -n schedulers/schedule_make_plots.sh
```

Develop and test locally, then push and deploy the exact clean renderer commit
with authorization. Use the Venus and command-center skills before a PBS run.
`schedulers/schedule_make_plots.sh` accepts `STAGE1_FILE`,
`TEMPERATURE_CLIMATOLOGY_FILE`, and `TAS_THRESHOLD_FILE`, in addition to the
existing dataset, time, data/output/log roots, and diagnostic options. Require
`PROJECT_ROOT` and `EXPECTED_COMMIT`. For a single-year results-only render use
one CPU, 4 GB RAM and 10 minutes; larger diagnostic bundles can override PBS
resources at submission. The scheduler uses `VENUS_MAMBA_ENV` (default
`dev_env`), a writable Matplotlib cache, and strict Python warnings.

Completion requires PBS F/Exit0, correct source identities and coverage,
warning-free logs, the expected nonempty decodable figure inventory, checksum
verification, and original-resolution review of every revised result figure.
