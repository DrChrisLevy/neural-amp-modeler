"""Measure NAM v3 timing pulses without importing the neural-network trainer."""

import numpy as np


def measure_v3_latency(wet):
    """Match NAM's onset thresholds and causal margin, checking both pulses.

    The 500 ms search also covers buffered USB recordings. Only the training
    calibration region is inspected; the held-out passages are not used.
    """
    lookahead, lookback = 1000, 24_000
    if np.ndim(wet) != 1 or len(wet) < 552_000 + lookback:
        raise ValueError("NAM v3 calibration requires a complete mono recording.")
    background = float(np.max(np.abs(wet[492_000:498_000])))
    threshold = max(background + 0.0003, 1.001 * background)
    scans = np.stack([wet[p - lookahead:p + lookback] for p in (504_000, 552_000)])
    delays = []
    for scan in scans:
        triggered = np.flatnonzero(np.abs(scan) > threshold)
        if not len(triggered):
            raise ValueError("NAM could not find both test pulses within 500 ms. Check the USB routing and levels.")
        delays.append(int(triggered[0]) - lookahead)
    triggered = np.flatnonzero(np.abs(scans.mean(axis=0)) > threshold)
    if not len(triggered):
        raise ValueError("NAM could not calibrate the test pulses. Their responses cancel each other.")
    estimated = int(triggered[0]) - lookahead
    if (min(delays) == -lookahead or max(delays) - min(delays) >= 20
            or any(abs(estimated - delay) >= 20 for delay in delays)):
        raise ValueError("NAM's test pulses do not have a consistent delay. Check noise, gates, and time effects.")
    return {"method": "nam_v3_blips", "estimated_delay_samples": estimated,
            "applied_delay_samples": estimated - 1, "retained_delay_samples": 1,
            "pulse_delays_samples": delays, "warnings": []}
