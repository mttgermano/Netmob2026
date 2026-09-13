# Full report

**Looking for the analysis code? It is in [`../code`](../code).**

The 2-page extended abstract links here, so this note explains what is and is
not in this directory.

## The report itself is not published here

Per the challenge terms the complete report remains **confidential**: the
organisers use it to select awardees and to decide whether the work is
presented as a talk or a poster. Its sources are therefore not in this
repository.

## Everything in it is reproducible

Given access to the dataset, every number in the report comes out of
[`../code`](../code). The scripts behind the main results are:

| Script | Result |
| :--- | :--- |
| `ablation_norain.py` | the rainfall circularity ablation |
| `threshold_calib.py` | per-line threshold calibration |
| `baseline_raw.py` | the raw-feature baselines |
| `exp_gatebias_leadlag.py` | quality-gate bias and the lead--lag check |
| `exp_e10_chronic.py` | chronic line-directions and concentration |
| `make_report_figs.py` | the report figures |

The dataset is released under an NDA to admitted challenge participants and is
not redistributed here. See the [root README](../README.md) for how to place
your own copy.
