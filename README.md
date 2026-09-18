# EHB paper plots

This plotting-only repository consumes immutable reduced EHB outputs. It does
not retrieve ERA5 or calculate heat budgets. Use the shared `dev_env` and the
calculation repository's dependency contract.

```bash
python scripts/make_plots.py --dataset DATASET --data-root INPUT_ROOT \
  --start 2021-06-01T00:00:00 --end 2021-08-31T23:00:00 \
  --plots-root NEW_OUTPUT_ROOT --diagnostic-plots --benchmark-diagnostics off
```

The input is `INPUT_ROOT/DATASET/heat_budget.nc`; retain its source manifest as
`run_info.json` beside it. Symlinks can adapt existing annual outputs without
duplicating data. Select a fresh output root to preserve previous figures.
Record calculation and plotting commits, input checksums, selected dates,
environment, command, and output checksums beside the generated figures.

This branch preserves the results and core diagnostics from `92721f3`, with
one presentation correction: the cumulative mass-residual panel uses the
default automatic tick locator, like the other panels, instead of a fixed
0.02 interval. This keeps labels legible for larger cumulative drifts and
provides useful ticks for small drifts. Shared numeric formatting and all
scientific curves, normalization, and integration remain unchanged. Preserve
exact `92721f3` renders separately when historical reproduction is required.

Validate in an explicitly activated `dev_env`:

```bash
python -c 'import sys; print(sys.executable)'
MPLCONFIGDIR=/tmp/ehb-paper-mpl python -W error -m pytest -q
python scripts/make_plots.py --help
bash -n schedulers/schedule_make_plots.sh
```

Use the Venus and command-center skills for scheduler runs. Submit the tracked
`schedulers/schedule_make_plots.sh` from a clean, commit-pinned checkout with
external data, plot, and log roots. Require successful PBS accounting,
warning-free logs, checksum verification, and visual inspection of real
figures before accepting publication output.
