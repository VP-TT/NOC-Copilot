"""
Synthetic telemetry generator with a GRADUAL precursor fault.

Why gradual matters: the whole point of this project is to *forecast* degradation
with enough lead time to intervene -- not to detect a failure that already happened.
A fault that ramps up over many timesteps produces a learnable precursor signature
and lets us measure lead time = (SLA breach time) - (first risk flag time).
An instant failure would give zero lead time and score nothing on the rubric.

This module is deliberately self-contained and dependency-light (numpy + pandas) so
it doubles as the day-one seed of the full build's ML track: swap the synthetic
series for real telemetry later, keep the same schema and ground-truth labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Reproducible: fixed seed so the demo and the recording always look the same.
SEED = 7

# Metric schema -- shared contract between generator, predictor, and copilot.
METRICS = ["link_utilization", "jitter_ms", "packet_loss_pct", "latency_ms", "bgp_flap_count"]

# SLA thresholds: a breach is declared when packet loss crosses this line.
SLA_PACKET_LOSS_PCT = 2.0


def generate_scenario(
    n_steps: int = 120,
    fault_start: int = 60,
    breach_step: int | None = None,
    site: str = "BR-07 (spoke) <-> HUB-1",
) -> tuple[pd.DataFrame, dict]:
    """
    Build the 'progressive hub-spoke congestion' scenario.

    Returns (dataframe, ground_truth):
      - dataframe: one row per timestep, columns = METRICS + 'minute'
      - ground_truth: {'fault_start', 'breach_step', 'site', 'scenario'}

    Timeline:
      [0, fault_start)        healthy baseline with realistic noise
      [fault_start, breach)   gradual ramp: utilization 40% -> 95%, jitter & loss trend up
      breach_step             packet loss crosses SLA_PACKET_LOSS_PCT  (ground-truth label)
    """
    rng = np.random.default_rng(SEED)
    t = np.arange(n_steps)

    # --- Healthy baselines (mean + small gaussian noise) ---
    link_util = np.full(n_steps, 40.0) + rng.normal(0, 1.5, n_steps)
    jitter = np.full(n_steps, 3.0) + rng.normal(0, 0.4, n_steps)
    loss = np.clip(rng.normal(0.05, 0.03, n_steps), 0, None)
    latency = np.full(n_steps, 28.0) + rng.normal(0, 1.0, n_steps)
    bgp_flap = np.zeros(n_steps)

    # --- Gradual precursor ramp from fault_start onward ---
    ramp_len = n_steps - fault_start
    ramp = np.linspace(0.0, 1.0, ramp_len)  # 0 -> 1 over the ramp window

    # Utilization creeps 40% -> ~95% (this is the leading indicator).
    link_util[fault_start:] = 40.0 + ramp * 55.0 + rng.normal(0, 1.5, ramp_len)
    # Jitter trends up as queues fill: 3ms -> ~14ms.
    jitter[fault_start:] = 3.0 + ramp * 11.0 + rng.normal(0, 0.5, ramp_len)
    # Latency rises with queue depth: 28ms -> ~70ms.
    latency[fault_start:] = 28.0 + ramp * 42.0 + rng.normal(0, 1.5, ramp_len)
    # Packet loss is the LAGGING/SLA metric: stays ~0 then accelerates near the end.
    loss[fault_start:] = (ramp ** 3) * 4.0 + rng.normal(0, 0.05, ramp_len)
    loss = np.clip(loss, 0, None)

    link_util = np.clip(link_util, 0, 100)

    df = pd.DataFrame(
        {
            "minute": t,
            "link_utilization": np.round(link_util, 2),
            "jitter_ms": np.round(jitter, 2),
            "packet_loss_pct": np.round(loss, 3),
            "latency_ms": np.round(latency, 2),
            "bgp_flap_count": bgp_flap.astype(int),
        }
    )

    # Ground-truth breach = first step where loss crosses the SLA line (after fault_start).
    if breach_step is None:
        crossed = df.index[(df["packet_loss_pct"] >= SLA_PACKET_LOSS_PCT) & (df["minute"] >= fault_start)]
        breach_step = int(crossed[0]) if len(crossed) else None

    ground_truth = {
        "scenario": "progressive_hub_spoke_congestion",
        "label": "Progressive hub-spoke congestion",
        "site": site,
        "fault_start": fault_start,
        "breach_step": breach_step,
        "sla_metric": "packet_loss_pct",
        "sla_threshold": SLA_PACKET_LOSS_PCT,
        # Scenario-specific knobs consumed by the predictor / copilot / dashboard.
        "lead_metric": "link_utilization",
        "lead_metric_label": "Link utilization (%)",
        "critical_level": 85.0,
        "slope_ref": 0.30,
        "at_risk_path": [("BR-07", "PE1"), ("PE1", "HUB-1")],
        "runbook_hint": "RB-CONGESTION-01",
    }
    return df, ground_truth


def generate_bgp_flap(
    n_steps: int = 120,
    fault_start: int = 60,
    breach_step: int | None = None,
    site: str = "BR-02 (spoke) <-> PE2",
) -> tuple[pd.DataFrame, dict]:
    """
    Build the 'BGP flap -> reroute cascade' scenario.

    An unstable PE-CE adjacency flaps with rising frequency; repeated best-path
    recomputation churns routes and latency on rerouted prefixes climbs until it
    breaches the latency SLA. Leading indicator = bgp_flap_count (link utilization
    stays nominal here -- a different precursor signature from the congestion case).
    """
    rng = np.random.default_rng(SEED + 1)
    t = np.arange(n_steps)

    # Utilization stays healthy in this scenario (the fault is routing, not load).
    link_util = np.full(n_steps, 38.0) + rng.normal(0, 1.5, n_steps)
    jitter = np.full(n_steps, 3.0) + rng.normal(0, 0.4, n_steps)
    loss = np.clip(rng.normal(0.05, 0.03, n_steps), 0, None)
    latency = np.full(n_steps, 28.0) + rng.normal(0, 1.0, n_steps)
    bgp_flap = np.zeros(n_steps, dtype=float)

    ramp_len = n_steps - fault_start
    ramp = np.linspace(0.0, 1.0, ramp_len)

    # Flap frequency is the LEADING indicator: 0 -> ~12 flaps/interval.
    bgp_flap[fault_start:] = ramp * 12.0 + rng.normal(0, 0.5, ramp_len)
    bgp_flap = np.clip(bgp_flap, 0, None)
    # Latency on rerouted prefixes accelerates with route churn: 28ms -> ~110ms (SLA metric).
    latency[fault_start:] = 28.0 + (ramp ** 2) * 90.0 + rng.normal(0, 1.5, ramp_len)
    # Minor transient loss during reroutes.
    loss[fault_start:] = (ramp ** 2) * 0.5 + rng.normal(0, 0.04, ramp_len)
    loss = np.clip(loss, 0, None)
    link_util = np.clip(link_util, 0, 100)

    df = pd.DataFrame(
        {
            "minute": t,
            "link_utilization": np.round(link_util, 2),
            "jitter_ms": np.round(jitter, 2),
            "packet_loss_pct": np.round(loss, 3),
            "latency_ms": np.round(latency, 2),
            "bgp_flap_count": np.round(bgp_flap).astype(int),
        }
    )

    sla_metric, sla_threshold = "latency_ms", 90.0
    if breach_step is None:
        crossed = df.index[(df[sla_metric] >= sla_threshold) & (df["minute"] >= fault_start)]
        breach_step = int(crossed[0]) if len(crossed) else None

    ground_truth = {
        "scenario": "bgp_flap_reroute_cascade",
        "label": "BGP flap -> reroute cascade",
        "site": site,
        "fault_start": fault_start,
        "breach_step": breach_step,
        "sla_metric": sla_metric,
        "sla_threshold": sla_threshold,
        "lead_metric": "bgp_flap_count",
        "lead_metric_label": "BGP flaps / interval",
        "critical_level": 8.0,
        "slope_ref": 0.10,
        "at_risk_path": [("BR-02", "PE2"), ("PE2", "HUB-1")],
        "runbook_hint": "RB-BGP-FLAP-02",
    }
    return df, ground_truth


def generate_mpls_underlay(
    n_steps: int = 120,
    fault_start: int = 60,
    breach_step: int | None = None,
    site: str = "HUB-1 <-> DC (MPLS core underlay)",
) -> tuple[pd.DataFrame, dict]:
    """
    Build the 'intermittent MPLS underlay loss' scenario.

    A flaky underlay segment degrades an LSP: tunnel keepalive JITTER trends up first
    (leading indicator), then intermittent packet-loss bursts grow until loss breaches
    the SLA. Utilization stays nominal -- the fault is transport, not load.
    """
    rng = np.random.default_rng(SEED + 2)
    t = np.arange(n_steps)

    link_util = np.full(n_steps, 42.0) + rng.normal(0, 1.5, n_steps)
    jitter = np.full(n_steps, 3.0) + rng.normal(0, 0.4, n_steps)
    loss = np.clip(rng.normal(0.05, 0.03, n_steps), 0, None)
    latency = np.full(n_steps, 30.0) + rng.normal(0, 1.0, n_steps)
    bgp_flap = np.zeros(n_steps, dtype=float)

    ramp_len = n_steps - fault_start
    ramp = np.linspace(0.0, 1.0, ramp_len)

    # Jitter is the LEADING indicator: 3ms -> ~18ms as the underlay destabilizes.
    jitter[fault_start:] = 3.0 + ramp * 15.0 + rng.normal(0, 0.6, ramp_len)
    # Latency wobbles upward with the flaky segment.
    latency[fault_start:] = 30.0 + ramp * 25.0 + rng.normal(0, 2.0, ramp_len)
    # Loss stays low then bursts up (SLA metric); bursts grow with severity.
    bursts = (rng.random(ramp_len) < 0.35) * rng.uniform(0, 1.0, ramp_len)
    loss[fault_start:] = (ramp ** 3) * 3.5 + bursts * ramp + rng.normal(0, 0.05, ramp_len)
    loss = np.clip(loss, 0, None)
    link_util = np.clip(link_util, 0, 100)

    df = pd.DataFrame(
        {
            "minute": t,
            "link_utilization": np.round(link_util, 2),
            "jitter_ms": np.round(jitter, 2),
            "packet_loss_pct": np.round(loss, 3),
            "latency_ms": np.round(latency, 2),
            "bgp_flap_count": bgp_flap.astype(int),
        }
    )

    sla_metric, sla_threshold = "packet_loss_pct", 2.0
    if breach_step is None:
        crossed = df.index[(df[sla_metric] >= sla_threshold) & (df["minute"] >= fault_start)]
        breach_step = int(crossed[0]) if len(crossed) else None

    ground_truth = {
        "scenario": "intermittent_mpls_underlay_loss",
        "label": "Intermittent MPLS underlay loss",
        "site": site,
        "fault_start": fault_start,
        "breach_step": breach_step,
        "sla_metric": sla_metric,
        "sla_threshold": sla_threshold,
        "lead_metric": "jitter_ms",
        "lead_metric_label": "Tunnel jitter (ms)",
        "critical_level": 15.0,
        "slope_ref": 0.18,
        "at_risk_path": [("HUB-1", "DC")],
        "runbook_hint": "RB-MPLS-UNDERLAY-03",
    }
    return df, ground_truth


def generate_policy_drift(
    n_steps: int = 120,
    fault_start: int = 60,
    breach_step: int | None = None,
    site: str = "PE2 policy scope (post controller push)",
) -> tuple[pd.DataFrame, dict]:
    """
    Build the 'controller misconfiguration / policy drift' scenario.

    A controller push mis-scopes a path, reshaping traffic onto one link: link
    UTILIZATION drifts upward (leading indicator) and rerouted-prefix LATENCY climbs
    until it breaches the latency SLA.
    """
    rng = np.random.default_rng(SEED + 3)
    t = np.arange(n_steps)

    link_util = np.full(n_steps, 38.0) + rng.normal(0, 1.4, n_steps)
    jitter = np.full(n_steps, 3.0) + rng.normal(0, 0.4, n_steps)
    loss = np.clip(rng.normal(0.05, 0.03, n_steps), 0, None)
    latency = np.full(n_steps, 28.0) + rng.normal(0, 1.0, n_steps)
    bgp_flap = np.zeros(n_steps, dtype=float)

    ramp_len = n_steps - fault_start
    ramp = np.linspace(0.0, 1.0, ramp_len)

    # Utilization drifts up as traffic is reshaped onto this link (leading indicator).
    link_util[fault_start:] = 38.0 + ramp * 42.0 + rng.normal(0, 1.4, ramp_len)
    # Latency on the mis-scoped path accelerates (SLA metric).
    latency[fault_start:] = 28.0 + (ramp ** 2) * 90.0 + rng.normal(0, 1.8, ramp_len)
    jitter[fault_start:] = 3.0 + ramp * 5.0 + rng.normal(0, 0.4, ramp_len)
    link_util = np.clip(link_util, 0, 100)

    df = pd.DataFrame(
        {
            "minute": t,
            "link_utilization": np.round(link_util, 2),
            "jitter_ms": np.round(jitter, 2),
            "packet_loss_pct": np.round(loss, 3),
            "latency_ms": np.round(latency, 2),
            "bgp_flap_count": bgp_flap.astype(int),
        }
    )

    sla_metric, sla_threshold = "latency_ms", 95.0
    if breach_step is None:
        crossed = df.index[(df[sla_metric] >= sla_threshold) & (df["minute"] >= fault_start)]
        breach_step = int(crossed[0]) if len(crossed) else None

    ground_truth = {
        "scenario": "controller_policy_drift",
        "label": "Controller misconfig -> policy drift",
        "site": site,
        "fault_start": fault_start,
        "breach_step": breach_step,
        "sla_metric": sla_metric,
        "sla_threshold": sla_threshold,
        "lead_metric": "link_utilization",
        "lead_metric_label": "Link utilization (%)",
        "critical_level": 78.0,
        "slope_ref": 0.25,
        "at_risk_path": [("PE2", "HUB-1")],
        "runbook_hint": "RB-POLICY-DRIFT-04",
    }
    return df, ground_truth


# --- Topology + scenario registry (consumed by the dashboard) ---

# Hand-placed node coordinates for a small representative MPLS WAN.
TOPOLOGY = {
    "nodes": {
        "BR-07": (0.0, 1.0),
        "BR-02": (0.0, 0.0),
        "PE1": (1.0, 1.0),
        "PE2": (1.0, 0.0),
        "HUB-1": (2.0, 0.5),
        "DC": (3.0, 0.5),
    },
    "edges": [
        ("BR-07", "PE1"),
        ("BR-02", "PE2"),
        ("PE1", "HUB-1"),
        ("PE2", "HUB-1"),
        ("HUB-1", "DC"),
    ],
}

SCENARIOS = {
    "congestion": {"label": "Progressive hub-spoke congestion", "generator": generate_scenario},
    "bgp_flap": {"label": "BGP flap -> reroute cascade", "generator": generate_bgp_flap},
    "mpls_underlay": {"label": "Intermittent MPLS underlay loss", "generator": generate_mpls_underlay},
    "policy_drift": {"label": "Controller misconfig -> policy drift", "generator": generate_policy_drift},
}


def build_scenario(key: str = "congestion") -> tuple[pd.DataFrame, dict]:
    """Return (dataframe, ground_truth) for a named scenario in the registry."""
    if key not in SCENARIOS:
        raise KeyError(f"unknown scenario '{key}'; choose from {list(SCENARIOS)}")
    return SCENARIOS[key]["generator"]()


if __name__ == "__main__":
    df, gt = generate_scenario()
    print(df.head())
    print("...")
    print(df.tail())
    print("ground_truth:", gt)
