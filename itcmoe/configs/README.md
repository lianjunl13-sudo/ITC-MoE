# Configurations

`sdar_c10.json`, `sdar_c20.json`, and `sdar_c30.json` select the base expert
compression budget, stage sequence, evaluation datasets and anchor coefficient.
Use an independent work directory per budget and a shared explicit
`--factor-library`. Pass `--reuse-calibration-from` for subsequent budgets to
reuse identical covariance and routed rank inputs.

`calibration_selection.json` contains 256 training-source IDs and content hashes.
It contains no dataset text or private paths. Other fixed mathematical settings
are documented in `docs/hyperparameters.md` and their source modules.
