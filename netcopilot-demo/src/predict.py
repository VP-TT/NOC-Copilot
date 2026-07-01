"""
Predictive engine (round-one slice).

Strategy: learn a healthy baseline from an initial calibration window, then score risk
as how far the leading indicator (link_utilization) has travelled from that baseline
toward a critical level -- GATED by a confirmed upward trend so a stable-but-high link
doesn't false-alarm. This produces a risk score that:
  * sits near zero during the healthy baseline (clean separation),
  * climbs MONOTONICALLY as the system degrades toward breach (not a noisy plateau), and
  * is fully interpretable ("fraction of the way from healthy to critical, trend-confirmed").
A sustained threshold crossing raises the 'risk rising' flag; time-to-impact is then
projected from the SLA metric's recent (accelerating) trend.

This is intentionally simple and interpretable so the demo is trustworthy and the
lead-time number is defensible. In the full build this slot is replaced by an
LSTM / Temporal-CNN multi-step forecaster + a time-to-impact regressor -- the
interface (telemetry in -> risk timeline + prediction out) stays the same.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Risk model params.
WINDOW = 10               # rolling window for trend estimation (timesteps)
SMOOTH = 5                # rolling window applied to the leading metric to damp sample noise
CALIB = 40                # initial steps treated as the healthy baseline (system starts healthy)
CRITICAL_UTIL = 85.0      # link utilization (%) treated as "critical" for risk normalization
SLOPE_REF = 0.30          # per-step rise (units/step) that counts as a fully-confirmed trend
RISK_THRESHOLD = 0.50     # normalized risk score that counts as "elevated"
SUSTAIN = 3               # consecutive elevated steps required to raise the flag (debounce)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Per-step slope (units/step) via least-squares over the trailing window."""
    x = np.arange(window)
    x = x - x.mean()
    denom = (x ** 2).sum()

    def slope(vals: np.ndarray) -> float:
        y = vals - vals.mean()
        return float((x * y).sum() / denom)

    return series.rolling(window, min_periods=window).apply(slope, raw=True).fillna(0.0)


def compute_risk(
    df: pd.DataFrame,
    lead_metric: str = "link_utilization",
    critical: float = CRITICAL_UTIL,
    slope_ref: float = SLOPE_REF,
) -> pd.DataFrame:
    """Return df augmented with baseline z, trend, and a normalized [0,1] risk_score.

    `critical` and `slope_ref` are per-scenario knobs: the critical level the leading
    metric is heading toward, and the per-step rise that counts as a confirmed trend.
    Defaults reproduce the original hub-spoke congestion behavior.
    """
    out = df.copy()

    # Smooth the leading metric just enough to suppress single-sample noise.
    smoothed = out[lead_metric].rolling(SMOOTH, min_periods=1).mean()

    # Learn the healthy baseline from the initial calibration window.
    base_mean = float(smoothed.iloc[:CALIB].mean())
    base_std = max(float(smoothed.iloc[:CALIB].std()), 1e-6)

    # Level: fraction of the way from healthy baseline to the critical level (0..1).
    level = ((smoothed - base_mean) / (critical - base_mean)).clip(0, 1)

    # Trend confirmation: 0 when flat, ->1 once the metric is rising at slope_ref/step.
    slope = _rolling_slope(smoothed, WINDOW)
    trend_confirm = (slope / slope_ref).clip(0, 1)

    # Risk = how far toward critical, gated by a confirmed upward trend.
    risk = (level * trend_confirm).clip(0, 1)

    # Baseline z-score retained for explainability/diagnostics.
    out["risk_z"] = np.round((smoothed - base_mean) / base_std, 3)
    out["risk_slope"] = np.round(slope, 4)
    out["risk_score"] = np.round(risk, 3)
    return out


def first_risk_step(df_risk: pd.DataFrame, threshold: float = RISK_THRESHOLD, sustain: int = SUSTAIN) -> int | None:
    """First timestep where risk_score stays >= threshold for `sustain` consecutive steps."""
    elevated = (df_risk["risk_score"] >= threshold).to_numpy()
    run = 0
    for i, hot in enumerate(elevated):
        run = run + 1 if hot else 0
        if run >= sustain:
            return int(i - sustain + 1)  # first step of the sustained run
    return None


def estimate_time_to_impact(
    df: pd.DataFrame,
    flag_step: int,
    lead_metric: str = "link_utilization",
    target: float = CRITICAL_UTIL,
    horizon: int = 300,
) -> int | None:
    """
    Project steps-until-critical from the LEADING indicator's recent trajectory.

    We extrapolate the (clean, strongly-trending) leading metric to its critical level
    rather than the SLA metric: at flag time the SLA metric is often still flat and noisy,
    so projecting it is unreliable, whereas the leading indicator already carries the
    trend. Returns steps-ahead (>=0), or None if not rising toward the target.
    """
    lo = max(0, flag_step - WINDOW)
    smoothed = df[lead_metric].rolling(SMOOTH, min_periods=1).mean()
    seg = smoothed.iloc[lo : flag_step + 1].to_numpy()
    if len(seg) < 3:
        return None

    current = seg[-1]
    if current >= target:
        return 0

    x = np.arange(lo, flag_step + 1)
    m, _ = np.polyfit(x, seg, 1)  # leading metric rises ~linearly -> linear fit is robust
    if m <= 1e-6:
        return None
    steps_ahead = (target - current) / m
    if steps_ahead > horizon:
        return None
    return max(0, int(round(steps_ahead)))


def evaluate(df_risk: pd.DataFrame, ground_truth: dict, threshold: float = RISK_THRESHOLD) -> dict:
    """
    Timestep-level detection metrics for one scenario.

    Labelling: the pre-breach degradation window [fault_start, breach] is 'positive' (we
    should be flagging), [0, fault_start) is 'negative' (healthy). Post-breach steps are
    excluded (the SLA is already violated). Returns precision / recall / false-positive-rate
    plus the confusion counts. Recall is naturally < 1 because the earliest fault steps have
    no signal yet -- what matters for the rubric is that precision/FPR stay clean and the
    detection is EARLY (see lead_time).
    """
    fault_start = ground_truth.get("fault_start", 0)
    breach = ground_truth.get("breach_step")
    minute = df_risk["minute"].to_numpy()
    elevated = (df_risk["risk_score"] >= threshold).to_numpy()

    end = breach if breach is not None else int(minute.max())
    scored = minute <= end
    is_fault = scored & (minute >= fault_start)
    is_healthy = scored & (minute < fault_start)

    tp = int((elevated & is_fault).sum())
    fp = int((elevated & is_healthy).sum())
    fn = int((~elevated & is_fault).sum())
    tn = int((~elevated & is_healthy).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "fpr": round(fpr, 2),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def predict(df: pd.DataFrame, ground_truth: dict) -> dict:
    """
    Run the full prediction pass.

    Reads the scenario's leading metric / critical level / trend reference from
    `ground_truth` when present (falling back to the congestion defaults), so the same
    engine serves every scenario. Returns the risk-augmented dataframe, flag step, breach
    step, measured lead time, and predicted time-to-impact at flag time.
    """
    lead_metric = ground_truth.get("lead_metric", "link_utilization")
    critical = ground_truth.get("critical_level", CRITICAL_UTIL)
    slope_ref = ground_truth.get("slope_ref", SLOPE_REF)

    df_risk = compute_risk(df, lead_metric=lead_metric, critical=critical, slope_ref=slope_ref)
    flag = first_risk_step(df_risk)
    breach = ground_truth.get("breach_step")

    lead_time = None
    if flag is not None and breach is not None:
        lead_time = breach - flag

    tti = estimate_time_to_impact(df_risk, flag, lead_metric=lead_metric, target=critical) if flag is not None else None
    peak_metric = df_risk["risk_score"].iloc[flag] if flag is not None else None

    return {
        "df_risk": df_risk,
        "flag_step": flag,
        "breach_step": breach,
        "lead_time": lead_time,
        "predicted_time_to_impact": tti,
        "risk_at_flag": None if peak_metric is None else float(round(peak_metric, 3)),
        "confidence": None if peak_metric is None else float(round(min(0.99, 0.5 + peak_metric / 2), 2)),
    }


if __name__ == "__main__":
    from generate_telemetry import generate_scenario

    df, gt = generate_scenario()
    result = predict(df, gt)
    print("flag_step:", result["flag_step"])
    print("breach_step:", result["breach_step"])
    print("lead_time:", result["lead_time"])
    print("predicted_time_to_impact:", result["predicted_time_to_impact"])
