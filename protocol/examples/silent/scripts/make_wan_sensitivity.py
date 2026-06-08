#!/usr/bin/env python3
"""Generate the SILENT WAN sensitivity model used in the paper draft.

The model intentionally uses only measured online bytes and public schedule
events. It is not a replacement for Linux tc or real-cloud WAN measurements.
"""

from __future__ import annotations

import argparse
import csv
import sys


def latency_min(
    online_compute_s: float,
    online_gb: float,
    bandwidth_mbps: float,
    rtt_ms: float,
    availability_flights: int,
    offline_preprocess_s: float,
) -> tuple[float, float]:
    transport_s = online_gb * 8000.0 / bandwidth_mbps
    barrier_s = availability_flights * rtt_ms / 1000.0
    online_s = online_compute_s + transport_s + barrier_s
    return online_s / 60.0, (online_s + offline_preprocess_s) / 60.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-compute-s", type=float, default=21.18)
    parser.add_argument("--online-gb", type=float, default=2.10)
    parser.add_argument("--offline-preprocess-s", type=float, default=46.46)
    parser.add_argument("--availability-flights", type=int, default=87)
    parser.add_argument(
        "--rtts-ms",
        default="0.5,4,20,40,80,100",
        help="comma-separated RTT sweep at 400 Mbps",
    )
    parser.add_argument(
        "--bandwidths-mbps",
        default="100,400,1000",
        help="comma-separated bandwidth sweep at 40 ms RTT",
    )
    args = parser.parse_args()

    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "sweep",
            "setting",
            "bandwidth_mbps",
            "rtt_ms",
            "online_min",
            "unamortized_min",
        ]
    )

    for rtt in [float(value) for value in args.rtts_ms.split(",") if value]:
        online, unamortized = latency_min(
            args.online_compute_s,
            args.online_gb,
            400.0,
            rtt,
            args.availability_flights,
            args.offline_preprocess_s,
        )
        writer.writerow(["rtt_at_400mbps", f"{rtt:g}ms", 400.0, rtt, f"{online:.4f}", f"{unamortized:.4f}"])

    for bandwidth in [float(value) for value in args.bandwidths_mbps.split(",") if value]:
        online, unamortized = latency_min(
            args.online_compute_s,
            args.online_gb,
            bandwidth,
            40.0,
            args.availability_flights,
            args.offline_preprocess_s,
        )
        writer.writerow(["bandwidth_at_40ms", f"{bandwidth:g}Mbps", bandwidth, 40.0, f"{online:.4f}", f"{unamortized:.4f}"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
