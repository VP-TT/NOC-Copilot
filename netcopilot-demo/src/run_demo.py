"""
End-to-end demo orchestrator: generate -> predict -> explain -> chart.

Run from the project root:
    python src/run_demo.py

Produces (under outputs/):
    telemetry.csv        full risk-augmented time series
    risk_timeline.png    chart with the risk-flag marker BEFORE the breach marker
    copilot_card.json    structured, grounded remediation card

The console prints the headline lead-time number -- the single most important
quantity in the whole demo, and what makes the recording sell the concept.
"""

from __future__ import annotations

import json
import os
import sys

# Allow `python src/run_demo.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")  # headless: write PNG without a display
import matplotlib.pyplot as plt

from generate_telemetry import generate_scenario, SLA_PACKET_LOSS_PCT
from predict import predict, RISK_THRESHOLD
from copilot import generate_card

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")


def _plot(df_risk, flag_step, breach_step, path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Top: the metrics that tell the degradation story.
    ax1.plot(df_risk["minute"], df_risk["link_utilization"], label="Link utilization (%)", color="#1f77b4")
    ax1.plot(df_risk["minute"], df_risk["jitter_ms"], label="Jitter (ms)", color="#ff7f0e")
    ax1.plot(df_risk["minute"], df_risk["packet_loss_pct"] * 20, label="Packet loss (%) x20", color="#d62728")
    ax1.axhline(SLA_PACKET_LOSS_PCT * 20, ls=":", color="#d62728", alpha=0.6, label="SLA loss threshold")
    ax1.set_ylabel("metric value")
    ax1.set_title("Network telemetry — progressive hub-spoke congestion")
    ax1.legend(loc="upper left", fontsize=8)

    # Bottom: the risk score with the two markers that prove lead time.
    ax2.plot(df_risk["minute"], df_risk["risk_score"], label="Risk score", color="#2ca02c")
    ax2.axhline(RISK_THRESHOLD, ls="--", color="#2ca02c", alpha=0.5, label="Risk threshold")
    if flag_step is not None:
        ax2.axvline(flag_step, color="#2ca02c", lw=2, label=f"Risk flag (t={flag_step})")
    if breach_step is not None:
        ax2.axvline(breach_step, color="#d62728", lw=2, label=f"SLA breach (t={breach_step})")
    if flag_step is not None and breach_step is not None:
        ax2.axvspan(flag_step, breach_step, color="#2ca02c", alpha=0.12)
        mid = (flag_step + breach_step) / 2
        ax2.annotate(
            f"LEAD TIME = {breach_step - flag_step} steps",
            xy=(mid, 0.5), ha="center", fontsize=11, fontweight="bold", color="#11611c",
        )
    ax2.set_ylabel("risk score")
    ax2.set_xlabel("time (steps / minutes)")
    ax2.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Generate telemetry with a gradual precursor fault + ground truth.
    df, gt = generate_scenario()

    # 2) Predict: risk timeline, flag step, lead time, time-to-impact.
    result = predict(df, gt)
    df_risk = result["df_risk"]
    flag, breach, lead = result["flag_step"], result["breach_step"], result["lead_time"]

    # 3) Explain: structured, grounded copilot card.
    card = generate_card(result, gt)

    # 4) Persist artifacts.
    df_risk.to_csv(os.path.join(OUT_DIR, "telemetry.csv"), index=False)
    with open(os.path.join(OUT_DIR, "copilot_card.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    _plot(df_risk, flag, breach, os.path.join(OUT_DIR, "risk_timeline.png"))

    # 5) Headline summary for the screen recording.
    print("=" * 64)
    print("  OFFLINE PREDICTIVE NETWORK COPILOT — demo run")
    print("=" * 64)
    print(f"  Scenario        : {gt['scenario']}")
    print(f"  Affected site   : {gt['site']}")
    print(f"  Risk flag raised: step {flag}  (risk={result['risk_at_flag']}, conf={result['confidence']})")
    print(f"  SLA breach at   : step {breach}  ({gt['sla_metric']} >= {gt['sla_threshold']})")
    print("-" * 64)
    if lead is not None and lead > 0:
        print(f"  >>> LEAD TIME = {lead} steps of warning BEFORE the SLA breach <<<")
    else:
        print("  >>> No positive lead time (check fault ramp / thresholds) <<<")
    print(f"  Predicted time-to-impact at flag: {result['predicted_time_to_impact']} steps")
    print("-" * 64)
    print("  COPILOT CARD")
    print(f"    issue     : {card['predicted_issue']}")
    print(f"    root cause: {card['root_cause_hypothesis']}")
    print(f"    grounded  : {card['grounded']}   citation: {card['citation']}")
    print(f"    actions   : {card['recommended_actions'][0]}")
    print("=" * 64)
    print(f"  Artifacts written to: {OUT_DIR}")
    print("    - telemetry.csv")
    print("    - risk_timeline.png")
    print("    - copilot_card.json")


if __name__ == "__main__":
    main()
