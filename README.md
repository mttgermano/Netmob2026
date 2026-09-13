# NetMob 2026 Data Challenge — Niterói bus mobility

Detecting delayed bus trips and service anomalies from GPS telemetry using
autoencoder latent feature representations, transferring the pipeline of
Duncan and Chen (2023) from network-censorship detection to public transport.

## What is here

| Path | Contents |
| :--- | :--- |
| `code/` | The full analysis pipeline: preprocessing, autoencoders, baselines, and every experiment script behind the reported numbers. See `code/README.md`. |
| `full_report/` | A pointer to the code behind the confidential full report, which is not published here. |

## The dataset is not in this repository

The NetMob 2026 mobility, ticketing, and line data are released under an NDA to
participants admitted to the challenge. They are not redistributed here, and
neither are the preprocessing caches, which hold per-trip GPS sequences and trip
metadata derived directly from them.

To reproduce the pipeline you need your own copy of the dataset, obtained from
the organisers. Place it so that the tree looks like this:

```
<parent>/
├── data/
│   ├── mobility_data/          # <date>.csv GPS traces
│   ├── gtfs/                   # dated GTFS snapshots
│   └── auxiliar_data/          # meteorological_data.csv, ticketing
└── Netmob2026/                 # this repository
    └── code/
```

Then run the pipeline from `code/`, starting with `00_preprocess.ipynb`.

## Headline results

The two findings the work rests on are negative, and both are reproducible from
the artifacts in `code/artifacts_*`.

- The learned representation does **not** improve delay detection. Gradient
  boosting on 48 per-trip summary statistics reaches 0.846 ROC-AUC against
  0.838 for the best autoencoder setup, and the gap widens as the
  preprocessing filters are relaxed.
- Two of the three original anomaly validations were **circular**, because
  rainfall was simultaneously an autoencoder input and the external check.
  Retraining without it erases the rain association and an apparent citywide
  disruption on 23 March.

What survives is a day-level signal stable across three independently trained
autoencoders, driven mostly by day type, plus a practical result: re-scaling
anomaly scores within each bus line lifts the alarm rate on the worst degraded
day from 5.7% to 29.3%, which beats any change of representation tested.
