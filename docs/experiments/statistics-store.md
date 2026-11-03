# Optional statistics-store experiment

[Experiments](index.md)

This experiment adds a country–indicator–year store to a separately loaded Gemma-4-31B backbone with
bidirectional prompt attention. Its external loader and evaluated backbone artifacts are not shipped.
The public preparation and reading code lives in `experiments.statistics`; these results are not a
reproduction claim for an ordinary Transformers Gemma checkpoint.

The attribution-only subset contains 23 OWID series. The store learns to query known indicators and
abstain on missing keys or unknown series. A dev-fitted threshold chooses between the store and the
backbone's own multiple-choice answer.

| set | engine alone | engine + store | store used |
|---|---|---|---|
| held-out keys | 70.70 | 99.73 | 99.5% |
| held-out series | 66.77 | 66.77 | 0.0% |
| unknown series (the half not used for the fit) | 40.97 | 40.18 | 3.0% |

Stored but unasked keys benefit; series omitted from training do not. The full 46-series store also
confuses nearby indicators, such as one vaccine series with another. A threshold fitted on known
and wholly unknown series does not cover that failure mode.

[Detailed methods and results](appendices/statistics-details.md) preserve series selection and licences,
decontamination checks, negative sets, latency and false-positive measurements.
[External backbone loaders](../how-to/custom-backbones.md) describe the optional interface;
`python -m experiments.statistics.owid_store --help` lists public preparation steps.
