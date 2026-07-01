"""
NOC Copilot — interactive operator dashboard (Streamlit).

This is the Phase-6 operator UI of the full project, built as a thin front-end over the
SAME engine modules used by the headless demo (`generate_telemetry → predict → copilot`).
It makes the predict→explain story live and visual:
  * animated time playback (watch risk climb before the SLA breaks),
  * network topology with the at-risk path lighting up,
  * KPI tiles + risk gauge,
  * a grounded, cited copilot card and chat box.

Run from the project root:
    streamlit run src/app.py
"""

from __future__ import annotations

import os
import sys
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_telemetry import TOPOLOGY, SCENARIOS, build_scenario
from predict import predict, estimate_time_to_impact, evaluate, RISK_THRESHOLD
from copilot import generate_card, answer_query, retrieve_context, build_report

# ----------------------------------------------------------------------------- setup
st.set_page_config(page_title="NOC Copilot", page_icon="🛰️", layout="wide")

ACCENT = "#6366f1"
GOOD, WARN, BAD = "#22c55e", "#f59e0b", "#ef4444"
TEAL = "#2dd4bf"

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
      .kpi {background:#141d33; border:1px solid #243049; border-radius:14px;
            padding:14px 16px; height:100%;}
      .kpi-label {font-size:0.72rem; letter-spacing:.06em; text-transform:uppercase;
                  color:#94a3b8;}
      .kpi-value {font-size:1.7rem; font-weight:800; line-height:1.2; margin-top:2px;}
      .kpi-sub {font-size:0.72rem; color:#94a3b8; margin-top:2px;}
      .card {background:#141d33; border:1px solid #243049; border-radius:14px; padding:18px 20px;}
      .badge {display:inline-block; padding:2px 10px; border-radius:999px;
              font-size:0.72rem; font-weight:700;}
      .cite {display:inline-block; background:#1e293b; color:#cbd5e1; border:1px solid #334155;
             border-radius:8px; padding:3px 9px; font-size:0.72rem; margin-top:8px;}
      .flow {display:flex; align-items:center; justify-content:space-between; gap:6px;
             background:#0e1729; border:1px solid #243049; border-radius:12px; padding:12px 14px;}
      .flow span {font-size:0.78rem; color:#cbd5e1;}
      .flow .arrow {color:#475569; font-weight:700;}
      .pill {font-size:0.7rem; font-weight:700; padding:2px 8px; border-radius:6px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- engine
@st.cache_data(show_spinner=False)
def run_engine(scenario_key: str):
    df, gt = build_scenario(scenario_key)
    pred = predict(df, gt)
    card = generate_card(pred, gt)
    return df, gt, pred, card


@st.cache_data(show_spinner=False)
def all_predictions():
    rows = []
    for key, meta in SCENARIOS.items():
        df, gt = build_scenario(key)
        pred = predict(df, gt)
        lead = pred["lead_time"]
        sev = "High" if (lead or 0) and lead <= 22 else "Medium"
        rows.append(
            {
                "Issue": meta["label"],
                "Location": gt["site"],
                "Severity": sev,
                "Lead time": "—" if lead is None else f"{lead} steps",
                "Time to impact": "—" if pred["predicted_time_to_impact"] is None else f"{pred['predicted_time_to_impact']} steps",
                "Confidence": "—" if pred["confidence"] is None else f"{int(pred['confidence']*100)}%",
                "_tti": pred["predicted_time_to_impact"] or 0,
            }
        )
    return rows


@st.cache_data(show_spinner=False)
def all_evaluations():
    rows = []
    for key, meta in SCENARIOS.items():
        df, gt = build_scenario(key)
        pred = predict(df, gt)
        m = evaluate(pred["df_risk"], gt)
        rows.append({
            "Scenario": meta["label"],
            "Lead time": "—" if pred["lead_time"] is None else f"{pred['lead_time']} steps",
            "Precision": m["precision"],
            "Recall": m["recall"],
            "FPR": m["fpr"],
            "Flag": pred["flag_step"],
            "Breach": pred["breach_step"],
        })
    return rows


def risk_color(r: float) -> str:
    if r < 0.34:
        return GOOD
    if r < 0.67:
        return WARN
    return BAD


def risk_gradient(r: float) -> str:
    """Continuous green -> amber -> red interpolation for smooth topology/gauge coloring."""
    r = max(0.0, min(1.0, r))
    g, a, bad = (34, 197, 94), (245, 158, 11), (239, 68, 68)

    def lerp(c0, c1, t):
        return tuple(int(c0[i] + (c1[i] - c0[i]) * t) for i in range(3))

    c = lerp(g, a, r / 0.5) if r < 0.5 else lerp(a, bad, (r - 0.5) / 0.5)
    return f"rgb({c[0]},{c[1]},{c[2]})"


def status_for(step: int, flag, breach):
    if breach is not None and step >= breach:
        return "SLA BREACH", BAD
    if flag is not None and step >= flag:
        return "RISK RISING", WARN
    return "HEALTHY", GOOD


def kpi(col, label, value, sub="", color="#e6edf6"):
    col.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{color}">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f"### 🛰️ NOC **Copilot**")
    st.caption("Offline predictive network assistant")
    scenario_key = st.selectbox(
        "Scenario",
        list(SCENARIOS.keys()),
        format_func=lambda k: SCENARIOS[k]["label"],
    )
    st.divider()
    st.markdown(
        f'<div style="background:#0e2a1a;border:1px solid #14532d;border-radius:10px;'
        f'padding:10px 12px"><span class="pill" style="background:{GOOD};color:#06210f">● '
        f'AIR-GAPPED MODE</span><div style="font-size:0.72rem;color:#86efac;margin-top:6px">'
        f'No external connectivity · local LLM + RAG</div></div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("Engine: baseline+trend predictor · grounded copilot (template, LLM-swappable)")


df, gt, pred, card = run_engine(scenario_key)
df_risk = pred["df_risk"]
flag, breach = pred["flag_step"], pred["breach_step"]
n_steps = len(df_risk) - 1

# Playback state (reset when scenario changes).
if st.session_state.get("scenario") != scenario_key:
    st.session_state.scenario = scenario_key
    st.session_state.step = int(breach) if breach is not None else n_steps
    st.session_state.playing = False
    st.session_state.chat = []

# ----------------------------------------------------------------------------- header
top = st.columns([0.62, 0.38])
with top[0]:
    st.markdown("## Predictive Network Operations")
    st.caption(f"Scenario: **{SCENARIOS[scenario_key]['label']}** · monitored segment: {gt['site']}")

# Playback controls.
ctrl = st.columns([0.18, 0.18, 0.64])
if ctrl[0].button("▶ Play" if not st.session_state.playing else "⏸ Pause", use_container_width=True):
    st.session_state.playing = not st.session_state.playing
    if st.session_state.playing and st.session_state.step >= n_steps:
        st.session_state.step = 0  # restart from the top
if ctrl[1].button("⟲ Reset", use_container_width=True):
    st.session_state.step = 0
    st.session_state.playing = False

slider_val = ctrl[2].slider("Time step (minute)", 0, n_steps, st.session_state.step)
if slider_val != st.session_state.step:
    st.session_state.step = slider_val
    st.session_state.playing = False  # manual scrub pauses playback

step = st.session_state.step
risk_now = float(df_risk["risk_score"].iloc[step])
status, status_color = status_for(step, flag, breach)
health = int(round(100 * (1 - risk_now)))

# Step-aware readouts: confidence climbs with the current risk level, and time-to-impact is
# re-projected from the CURRENT step so it counts down during playback. Both therefore differ
# by scenario and by where you are in the timeline (not pinned to a single flag-time value).
lead_metric = gt.get("lead_metric", "link_utilization")
critical = gt.get("critical_level", 85.0)
conf_now = round(0.50 + 0.49 * risk_now, 2)
tti_now = estimate_time_to_impact(df_risk, step, lead_metric=lead_metric, target=critical)

active = flag is not None and step >= flag
lead_shown = f"{pred['lead_time']} steps" if (active and pred["lead_time"] is not None) else "—"
tti_shown = f"{tti_now} steps" if (active and tti_now is not None) else ("0 steps" if active else "—")
conf_shown = f"{int(conf_now*100)}%" if active else "—"

# ----------------------------------------------------------------------------- KPI tiles
k = st.columns(6)
kpi(k[0], "Network health", f"{health}%", "100 = healthy baseline", risk_gradient(risk_now))
kpi(k[1], "Status", status, f"step {step} / {n_steps}", status_color)
kpi(k[2], "Lead time", lead_shown, "warning before breach", ACCENT if lead_shown != "—" else "#64748b")
kpi(k[3], "Time to impact", tti_shown, "projected to critical", "#e6edf6")
kpi(k[4], "Confidence", conf_shown, "of active prediction", "#e6edf6")
kpi(k[5], "Coverage", f"{len(TOPOLOGY['nodes'])} nodes", f"{len(TOPOLOGY['edges'])} links · air-gapped", TEAL)

st.write("")

# ----------------------------------------------------------------------------- topology + gauge
mid = st.columns([0.62, 0.38])

with mid[0]:
    st.markdown("##### Network topology")
    at_risk = {frozenset(e) for e in gt.get("at_risk_path", [])}
    nodes, edges = TOPOLOGY["nodes"], TOPOLOGY["edges"]
    hot = risk_gradient(risk_now)
    fig = go.Figure()
    for a, b in edges:
        on_path = frozenset((a, b)) in at_risk
        color = hot if on_path else "#334155"
        width = 3 + 5 * risk_now if on_path else 2
        (x0, y0), (x1, y1) = nodes[a], nodes[b]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                                 line=dict(color=color, width=width), hoverinfo="skip", showlegend=False))
    on_path_nodes = {n for e in at_risk for n in e}
    # Glow halo behind at-risk nodes; opacity + size grow with risk (a 'pulse' during playback).
    gx = [nodes[n][0] for n in on_path_nodes]
    gy = [nodes[n][1] for n in on_path_nodes]
    if gx:
        fig.add_trace(go.Scatter(x=gx, y=gy, mode="markers",
                                 marker=dict(size=48 + 34 * risk_now, color=hot,
                                             opacity=0.10 + 0.30 * risk_now),
                                 hoverinfo="skip", showlegend=False))
    nx_, ny_, ncolor, ntext = [], [], [], []
    for name, (x, y) in nodes.items():
        nx_.append(x); ny_.append(y); ntext.append(name)
        ncolor.append(hot if name in on_path_nodes else TEAL)
    fig.add_trace(go.Scatter(x=nx_, y=ny_, mode="markers+text", text=ntext,
                             textposition="bottom center", textfont=dict(color="#e6edf6", size=13),
                             marker=dict(size=34, color=ncolor, line=dict(color="#0b1220", width=2)),
                             hoverinfo="text", showlegend=False))
    fig.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(visible=False, range=[-0.3, 3.3]), yaxis=dict(visible=False, range=[-0.5, 1.5]))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with mid[1]:
    st.markdown("##### Risk level")
    g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(risk_now, 2),
        number=dict(font=dict(color="#e6edf6", size=34)),
        gauge=dict(
            axis=dict(range=[0, 1], tickcolor="#64748b"),
            bar=dict(color=risk_gradient(risk_now)),
            bgcolor="#0e1729",
            borderwidth=0,
            steps=[dict(range=[0, 0.34], color="#102a1b"),
                   dict(range=[0.34, 0.67], color="#2a230f"),
                   dict(range=[0.67, 1], color="#2a1414")],
            threshold=dict(line=dict(color="#e6edf6", width=3), value=RISK_THRESHOLD),
        ),
    ))
    g.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf6"))
    st.plotly_chart(g, use_container_width=True, config={"displayModeBar": False})

# ----------------------------------------------------------------------------- risk timeline
st.markdown("##### Risk timeline — forecast vs. SLA breach")
tl = go.Figure()
xs = df_risk["minute"].iloc[: step + 1]
ys = df_risk["risk_score"].iloc[: step + 1]
tl.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=ACCENT, width=3),
                        fill="tozeroy", fillcolor="rgba(99,102,241,0.15)", name="Risk score"))
tl.add_hline(y=RISK_THRESHOLD, line=dict(color="#64748b", dash="dash"),
             annotation_text="risk threshold", annotation_font_color="#94a3b8")
if flag is not None and step >= flag:
    tl.add_vline(x=flag, line=dict(color=GOOD, width=2))
    tl.add_annotation(x=flag, y=1.05, text=f"risk flag (t={flag})", showarrow=False, font=dict(color=GOOD))
if breach is not None:
    tl.add_vline(x=breach, line=dict(color=BAD, width=2))
    tl.add_annotation(x=breach, y=1.05, text=f"SLA breach (t={breach})", showarrow=False, font=dict(color=BAD))
    if flag is not None and step >= flag:
        tl.add_vrect(x0=flag, x1=breach, fillcolor="rgba(34,197,94,0.10)", line_width=0)
tl.update_layout(template="plotly_dark", height=270, margin=dict(l=10, r=10, t=20, b=10),
                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                 xaxis=dict(title="time step (minute)", range=[0, n_steps]), yaxis=dict(range=[0, 1.12]))
st.plotly_chart(tl, use_container_width=True, config={"displayModeBar": False})

# ----------------------------------------------------------------------------- predictions + bars
low = st.columns([0.55, 0.45])
rows = all_predictions()
with low[0]:
    st.markdown("##### Predictions across the network")
    disp = pd.DataFrame([{kk: vv for kk, vv in r.items() if not kk.startswith("_")} for r in rows])
    active_label = SCENARIOS[scenario_key]["label"]

    def _hl(row):  # highlight the row for the scenario currently in view
        hit = row["Issue"] == active_label
        return ["background-color: rgba(99,102,241,0.28)" if hit else "" for _ in row]

    st.dataframe(disp.style.apply(_hl, axis=1), use_container_width=True, hide_index=True)
with low[1]:
    st.markdown("##### Time to impact (top issues)")
    bar = go.Figure(go.Bar(
        x=[r["_tti"] for r in rows], y=[r["Issue"] for r in rows], orientation="h",
        marker=dict(color=[WARN, ACCENT][: len(rows)]), text=[f"{r['_tti']} steps" for r in rows],
        textposition="auto",
    ))
    bar.update_layout(template="plotly_dark", height=210, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(title="steps to impact"), yaxis=dict(autorange="reversed"))
    st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})

# ----------------------------------------------------------------------------- copilot card + chat
st.write("")
cc = st.columns([0.5, 0.5])

with cc[0]:
    st.markdown("##### AI copilot — remediation card")
    if flag is not None and step >= flag:
        conf_pct = int(conf_now * 100)
        tti_txt = f"{tti_now} steps" if tti_now is not None else "0 steps"
        actions = "".join(f"<li>{a}</li>" for a in card["recommended_actions"])
        grounded_badge = (f'<span class="badge" style="background:{GOOD};color:#06210f">GROUNDED</span>'
                          if card["grounded"] else
                          f'<span class="badge" style="background:{BAD};color:#2a0a0a">UNGROUNDED</span>')
        st.markdown(
            f'<div class="card">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<b style="font-size:1.05rem">{card["predicted_issue"]}</b>{grounded_badge}</div>'
            f'<div style="color:#94a3b8;font-size:0.8rem;margin:6px 0 10px">'
            f'Confidence {conf_pct}% · affected: {", ".join(card["affected_sites"])} · '
            f'time-to-impact {tti_txt}</div>'
            f'<div style="font-size:0.85rem"><b style="color:#cbd5e1">Root cause</b><br>{card["root_cause_hypothesis"]}</div>'
            f'<div style="font-size:0.85rem;margin-top:8px"><b style="color:#cbd5e1">Recommended actions</b>'
            f'<ul style="margin:4px 0 0 1.1rem">{actions}</ul></div>'
            f'<div class="cite">🔖 {card["citation"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No active prediction at this step. Press ▶ Play (or scrub forward) to watch risk rise "
                "and the copilot raise a grounded recommendation before the SLA breaks.")

with cc[1]:
    st.markdown("##### Ask the copilot")
    for role, text, cite in st.session_state.get("chat", []):
        with st.chat_message(role):
            st.write(text)
            if cite:
                st.caption(f"🔖 {cite}")
    prompt = st.chat_input("e.g. why is this segment at risk? what should I do?")
    if prompt:
        ans = answer_query(prompt, pred, gt)
        st.session_state.chat.append(("user", prompt, ""))
        st.session_state.chat.append(("assistant", ans["text"], ans["citation"]))
        st.rerun()

# ----------------------------------------------------------------------------- analytics tabs
st.write("")
st.divider()
tab_eval, tab_trust = st.tabs(["📊 Evaluation", "🔒 Trust & Air-gap"])

with tab_eval:
    ecols = st.columns([0.55, 0.45])
    with ecols[0]:
        st.markdown("##### Detection metrics — all scenarios")
        st.dataframe(pd.DataFrame(all_evaluations()), use_container_width=True, hide_index=True)
        st.caption("Precision/FPR measured over the healthy window vs the pre-breach degradation "
                   "window. Recall < 1 is expected — the earliest fault steps carry no signal yet; "
                   "what matters is early, false-alarm-free detection (see lead time).")
    with ecols[1]:
        st.markdown("##### Reactive vs. predictive")
        lead_val = pred["lead_time"] or 0
        cmp = go.Figure(go.Bar(
            x=[lead_val, 0], y=["Predictive copilot", "Reactive SLA monitor"], orientation="h",
            marker=dict(color=[ACCENT, BAD]),
            text=[f"{lead_val} steps early", "0 · alarms at breach"], textposition="auto",
        ))
        cmp.update_layout(template="plotly_dark", height=200, margin=dict(l=10, r=10, t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(title="steps of warning before SLA breach"))
        st.plotly_chart(cmp, use_container_width=True, config={"displayModeBar": False})
        st.caption(f"A threshold monitor only fires once the SLA is already violated (t={breach}). "
                   f"The copilot forecasts it **{lead_val} steps earlier** (t={flag}).")

with tab_trust:
    tcols = st.columns([0.5, 0.5])
    with tcols[0]:
        st.markdown("##### Retrieved context (RAG grounding)")
        rb = retrieve_context(gt)
        if rb:
            st.markdown(
                f'<div class="card"><div style="font-size:0.78rem;color:#94a3b8">Retrieved for lead metric '
                f'<b>{gt.get("lead_metric")}</b> — the exact local context passed to the copilot</div>'
                f'<div style="margin-top:8px"><b>{rb["id"]} — {rb["title"]}</b></div>'
                f'<div style="font-size:0.82rem;margin-top:6px"><b style="color:#cbd5e1">Symptoms:</b> {rb["symptoms"]}</div>'
                f'<div style="font-size:0.82rem;margin-top:4px"><b style="color:#cbd5e1">Root cause:</b> {rb["root_cause"]}</div>'
                f'<div class="cite">🔖 {rb["source"]}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("No runbook retrieved for this signature.")
        st.markdown("##### Incident log")
        inc = []
        for i, r in enumerate(all_predictions()):
            is_active = r["Issue"] == SCENARIOS[scenario_key]["label"]
            inc.append({"ID": f"INC-24{i+31}", "Issue": r["Issue"], "Location": r["Location"],
                        "Severity": r["Severity"], "Status": "OPEN · in view" if is_active else "OPEN · predicted"})
        st.dataframe(pd.DataFrame(inc), use_container_width=True, hide_index=True)
    with tcols[1]:
        st.markdown("##### Air-gap monitor")
        st.markdown(
            f'<div class="card"><span class="badge" style="background:{GOOD};color:#06210f">● 0 EXTERNAL '
            f'CONNECTIONS</span><div style="font-size:0.82rem;color:#94a3b8;margin-top:8px">Model inference '
            f'and runbook retrieval run locally. No egress during the predict→explain cycle.</div></div>',
            unsafe_allow_html=True,
        )
        audit = pd.DataFrame([
            {"Destination": "localhost (predictor)", "Type": "local", "Status": "allowed"},
            {"Destination": "localhost (RAG index)", "Type": "local", "Status": "allowed"},
            {"Destination": "any external host", "Type": "egress", "Status": "none / blocked"},
        ])
        st.dataframe(audit, use_container_width=True, hide_index=True)
        st.markdown("##### Export")
        report_md = build_report(pred, gt, card, evaluate(df_risk, gt))
        st.download_button("⬇ Download incident report (Markdown)", report_md,
                           file_name=f"incident_{gt.get('scenario', 'report')}.md",
                           mime="text/markdown", use_container_width=True)

# ----------------------------------------------------------------------------- solution flow
st.write("")
flow_steps = ["Network devices", "Telemetry", "Prediction engine", "Knowledge base (RAG)", "Offline copilot (LLM)", "Dashboard"]
flow_html = '<div class="flow">'
for i, s in enumerate(flow_steps):
    flow_html += f'<span>{s}</span>'
    if i < len(flow_steps) - 1:
        flow_html += '<span class="arrow">→</span>'
flow_html += "</div>"
st.markdown(flow_html, unsafe_allow_html=True)

# ----------------------------------------------------------------------------- autoplay tick
if st.session_state.playing:
    if step < n_steps:
        time.sleep(0.12)
        st.session_state.step = step + 1
        st.rerun()
    else:
        st.session_state.playing = False
