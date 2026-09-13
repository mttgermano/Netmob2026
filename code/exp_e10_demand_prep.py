"""E10 Part 1a — aggregate the 567 MB of March-2026 ticketing CSVs into a small
line x direction-agnostic demand table joinable to the trip table.

The ticket records carry vehicle_number, but those ids share NOTHING with the
vehicle ids in the AVL trip table (zero overlap on 2026-03-11), so a per-trip
join through the vehicle is impossible; the two datasets were anonymised
independently. The only usable key is the route code (route_name in tickets,
line_id in trips) plus the timestamp. Direction is absent from the ticket
schema, so demand is necessarily direction-agnostic — a real limitation, since
Part 2 shows delay is strongly direction-asymmetric.

Aggregation is therefore per (service_date, line, hour): boardings, distinct
riders, transfer share, free-pass share, cash share, mean fare. We also keep a
network-wide per-(date, hour) total so that a trip can be told how busy the
whole system is, not just its own line. Files are read one day at a time with
only the six needed columns, so peak memory stays near a single day's frame.

Route codes are normalised by stripping a trailing ".0" (the trip table stores
some line ids as floats) and upper-casing.

Output -> artifacts_e10/demand_line_hour.csv (small) and demand_net_hour.csv
"""
import glob
import os
import numpy as np
import pandas as pd
import netmob_prep

OUT = "./artifacts_e10"
COLS = ["transaction_date", "route_name", "integration_flag", "fare_type",
        "card_type", "anon_user_id"]


def norm_route(s):
    s = s.astype(str).str.strip().str.upper()
    return s.str.replace(r"\.0$", "", regex=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(netmob_prep.DATA_DIR, "ticket_data", "*.csv")))
    print(f"{len(files)} ticket files")
    line_parts, net_parts = [], []
    for f in files:
        t = pd.read_csv(f, usecols=COLS)
        ts = pd.to_datetime(t["transaction_date"], utc=True).dt.tz_convert("America/Sao_Paulo")
        t["service_date"] = ts.dt.strftime("%Y-%m-%d")
        t["hour"] = ts.dt.hour
        t["line_id"] = norm_route(t["route_name"])
        t["is_transfer"] = (t["integration_flag"] == 2).astype(np.int8)
        t["is_free"] = (t["fare_type"] == 1).astype(np.int8)
        t["is_cash"] = (t["fare_type"] == 3).astype(np.int8)
        g = t.groupby(["service_date", "line_id", "hour"], observed=True)
        line_parts.append(pd.DataFrame({
            "boardings": g.size(),
            "riders": g["anon_user_id"].nunique(),
            "transfer_share": g["is_transfer"].mean(),
            "free_share": g["is_free"].mean(),
            "cash_share": g["is_cash"].mean(),
        }).reset_index())
        gn = t.groupby(["service_date", "hour"], observed=True)
        net_parts.append(pd.DataFrame({"net_boardings": gn.size()}).reset_index())
        print(f"  {os.path.basename(f)}  {len(t):,} rows")
        del t, ts, g, gn

    line = pd.concat(line_parts, ignore_index=True)
    net = pd.concat(net_parts, ignore_index=True)
    line.to_csv(f"{OUT}/demand_line_hour.csv", index=False)
    net.to_csv(f"{OUT}/demand_net_hour.csv", index=False)
    print(f"\nline-hour cells {len(line):,}  total boardings {line.boardings.sum():,}")
    print(f"saved -> {OUT}/demand_line_hour.csv")

    trips = pd.read_parquet("./cache/trips_meta.parquet")
    tl = set(norm_route(trips["line_id"]))
    dl = set(line["line_id"])
    print(f"trip lines {len(tl)}  ticket lines {len(dl)}  matched {len(tl & dl)}")
    print("unmatched trip lines:", sorted(tl - dl))
    cov = trips.assign(l=norm_route(trips["line_id"]))["l"].isin(dl).mean()
    print(f"share of trips on a line with ticket data: {cov:.3%}")


if __name__ == "__main__":
    main()
