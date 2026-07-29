# EHB paper-plots agent instructions

This repository creates static publication figures from reduced Eulerian Heat
Budget datasets. It does not calculate the heat budget, retrieve ERA5 data, or
own the scientific formulation used to produce its inputs.

## Establish authority and scope

1. Inspect the branch, upstream, commit, worktree, and input provenance before
   changing code or preparing a plot run.
2. Keep busy or dirty checkouts untouched and use a separate local worktree.
3. Treat the corresponding EHB calculation repository and its scientific
   documentation as authoritative for variable definitions, units, signs, and
   closure interpretation.
4. Treat this repository as authoritative only for reduced-data loading,
   publication plotting behavior, plot style, and plot-run validation.

Do not compensate for missing or scientifically inconsistent inputs by
silently redefining a variable, changing a sign, fabricating data, or embedding
calculation logic in the plotting repository. Report the upstream discrepancy
and fix it at the correct source.

## Preserve project boundaries

- Keep reusable data loading and validation in `src/io.py`.
- Keep CLI parsing and orchestration in `src/cli.py` and
  `scripts/make_plots.py`.
- Keep visual constants and shared rendering behavior in `src/plot_style.py`.
  Plot modules must use that shared style rather than introducing local,
  figure-specific defaults without a documented reason.
- Keep result and diagnostic plot responsibilities separated in
  `src/plot_results.py` and `src/plot_diagnostics.py`.
- Keep scheduler integration in `schedulers/`; it must not redefine scientific
  variables or plotting semantics.
- Treat reduced NetCDF inputs and `run_info.json` as immutable upstream
  artifacts. Never modify an input dataset in place.
- Keep input data, generated plots, logs, caches, and scratch products outside
  Git unless the user explicitly approves a small publication artifact or test
  fixture for version control.
- Do not manually edit generated figures or generated metadata. Change the
  source data, plotting code, or declared configuration and regenerate them.
- Keep data and plot roots configurable through CLI arguments or environment
  variables. Do not add personal absolute paths to source or scheduler files.

## Preserve input and figure provenance

- Record the dataset identity, input path, available `run_info.json`, source
  EHB branch and commit, campaign or run ID, scientific configuration, selected
  time range, plotting branch and commit, environment, and output path.
- Reject dataset names containing path traversal and fail clearly when required
  variables, time coordinates, coverage, units, or metadata are missing.
- Keep benchmark diagnostics explicit. Do not enable or suppress them merely
  from a convenient filename when the requested scientific comparison is
  ambiguous.
- Keep plots deterministic for the same reduced dataset, configuration,
  dependency environment, and source commit.
- Preserve output separation by dataset and by result versus diagnostic
  figures. Do not overwrite an unrelated dataset's publication directory.

## Use the shared local environment

- Use the local Mamba environment named `dev_env` for compilation, tests, CLI
  checks, and small plotting prototypes.
- Activate it explicitly and print `sys.executable` before recording
  validation. Do not silently use the base environment.
- Treat the EHB dependency contract as authoritative until this repository
  gains its own tracked environment file. If its plotting dependencies diverge,
  add and document a reproducible project-local environment rather than relying
  on workstation state.
- Use a writable Matplotlib configuration directory for automated tests and
  remote jobs.

Minimal local validation is:

```bash
mamba activate dev_env
python -c "import sys; print(sys.executable)"
python -m pytest -q
python scripts/make_plots.py --help
```

Run plotting tests with warnings treated as errors. A warning-free synthetic
test does not replace opening and validating the intended real reduced
dataset.

## Route Venus and OpenPBS work

- Invoke the `venus-hpc` skill before deploying, submitting, monitoring, or
  diagnosing a Venus plot run.
- Develop and validate locally, commit intentionally, push the exact source
  commit, and fast-forward only a clean Venus checkout.
- Never run an agent or edit source on Venus. Never switch, fetch, pull, reset,
  clean, or stash a checkout referenced by a queued or running job.
- Submit plotting through OpenPBS rather than running it on the Venus login
  node.
- Require `PROJECT_ROOT` and `EXPECTED_COMMIT`; verify a clean checkout at the
  expected commit before plotting.
- Initialize `dev_env` explicitly and print the resolved interpreter.
- Require configurable data, plot, and log roots outside the source checkout.
- Use a short queued smoke run for a new dataset or plotting pathway before a
  longer publication run.
- Require authorization before pushing, deploying, transferring data,
  submitting or altering jobs, deleting outputs, or changing permissions.

## Track long-running work

- Invoke the `command-center` skill for qualifying asynchronous plot runs.
- In addition to the shared schema, record the dataset, intended figure set,
  plotting branch and commit, source EHB identity, selected period, resources,
  log path, and output directory.
- Mark the task complete only after terminal success, non-empty figure
  creation, expected figure inventory, warning-free logs, and visual review of
  the real artifacts.

## Validate figures as scientific artifacts

- Run focused tests during development and the complete suite before handoff.
  Resolve lint, test, plotting-warning, and flaky failures even when they
  predate the current change.
- Run `bash -n` on changed scheduler scripts and compile changed Python
  modules.
- Test missing variables, invalid dataset names, empty time selections,
  benchmark-mode dispatch, output path construction, and actual figure writes
  using small synthetic fixtures.
- For real-data validation, inspect required variables, dimensions, units,
  signs, coverage, selected time bounds, and input provenance before rendering.
- Inspect generated images directly. Check labels, legends, units, scales,
  date ticks, panel ordering, clipping, whitespace, line visibility, and
  consistency with the shared plot style.
- Treat successful file creation as necessary but not sufficient evidence that
  a publication figure is correct.

## Definition of done

A change is complete when:

1. Plotting behavior and input expectations are documented and tested.
2. Input scientific identities and plotting provenance are recorded.
3. Runtime paths are configurable and generated data remain outside Git.
4. Focused checks, the full suite, CLI help, and scheduler syntax pass in the
   declared environment without warnings.
5. Representative real figures are visually inspected when output appearance
   is part of the claim.
6. The final diff contains only intentional project files and any long-running
   plot run has a current command-center entry.
