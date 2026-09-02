# NetMob 2026 — Latent Feature Representation Learning on Niterói Bus Mobility

Detecting **delayed bus trips** and **service anomalies** from GPS telemetry
using autoencoder latent features (approach inspired by Duncan & Chen, 2023,
*Detecting network-based internet censorship via latent feature representation
learning*).

## Pipeline

| Step | File | Output |
| :--- | :--- | :--- |
| 0 | `00_preprocess.ipynb` (+ `netmob_prep.py`) | `cache/sequences.npz`, `cache/trips_meta.parquet` |
| 1 | `01_data_analysis.ipynb` | EDA figures in `cache/` |
| 2a | `ML_autoencoder_cnn.ipynb` | `artifacts_ae_cnn/` (metrics, curves, latents, anomaly scores) |
| 2b | `ML_autoencoder_lstm.ipynb` | `artifacts_ae_lstm/` |
| 3 | `03_anomaly_analysis.ipynb` | `artifacts_anomaly/` |

**Preprocessing** joins each GPS trip instance to its GTFS schedule
(mobility `tripId` = GTFS `trip_id` minus the trailing `_<direction>` suffix,
matched per direction and dated snapshot), map-matches points onto the route
shape, interpolates stop arrival times, and labels a trip **delayed** when its
median stop delay is ≥ 5 minutes. Each trip becomes a `(T=64, F=12)` sequence
of 1-minute-bin features (speed, acceleration, heading change, route progress,
cross-track error, dwell fraction, time-of-day, weekend flag, rain, headway).

**Modeling** (per AE variant): a Conv1D or Seq2Seq-LSTM autoencoder is trained
unsupervised per CV fold; the frozen encoder's 64-dim latents feed six
classifiers (MLP, XGBoost, RF, LR, LSTM, GRU). Cross-validation is
`StratifiedGroupKFold` grouped by **service date**, so models are always
evaluated on unseen days. Out-of-fold reconstruction errors double as
**anomaly scores**, validated in step 3 against known degraded-telemetry days
(Mar 20/22/28) and rainfall.

## Running

```bash
uv sync                                # Python 3.13, torch CPU wheels
# execute notebooks in order (Jupyter, VS Code, or nbclient)
```

Set `NETMOB_SMOKE=1` before running an `ML_autoencoder_*` notebook for a quick
sanity pass (3k trips, 3 epochs, 2 folds).

Raw data is expected in `../data/` (mobility, GTFS, ticket, auxiliar, census)
as distributed by the NetMob 2026 challenge; `cache/` and `artifacts_*/` are
generated and gitignored.
