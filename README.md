<h1 align="center">🛰️ NOC Copilot</h1>
<p align="center"><b>An offline, predictive network operations copilot — forecast degradation <i>before</i> the SLA breaks, and explain it with a grounded, cited AI assistant.</b></p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="Offline" src="https://img.shields.io/badge/Runs-100%25%20Offline-22c55e">
  <img alt="Air-gapped" src="https://img.shields.io/badge/Air--gapped-0%20egress-ef4444">
  <img alt="Event" src="https://img.shields.io/badge/ISRO-BAH%202026-ff7a1a">
</p>

<p align="center"><i>“Forecast, don't just detect.”</i></p>

---

## 📖 Overview

Mission-critical, air-gapped WANs (MPLS + IPSec overlays) fail in ways that **cascade** — a
congested hub, a flapping BGP session — and today's NOC tooling only **alerts after the SLA is
already breached**. That's reactive: operators clean up, they don't pre-empt.

**NOC Copilot** turns live telemetry into a **rising risk score with measurable lead time**, then
hands the operator a **structured, cited remediation card** — all running **100% offline**, which is
non-negotiable for sensitive/air-gapped environments.

> Built for the **ISRO Bhartiya Antariksh Hackathon (BAH) 2026** — round-one idea submission.
> This repository is a **working vertical slice**: real engine, real dashboard, real numbers.

---

## 🎬 Demo Video


**Youtube LINK to view Demo** - https://youtu.be/O8VrDtAuETs


**What the video shows:** press **Play** → watch risk climb as the network degrades → the at-risk path
lights up and the **risk flag fires before the SLA breach** → the copilot emits a **grounded, cited**
remediation card → ask it a question in plain English.

---

## ✨ Why it stands out

| | |
|---|---|
| ⏱️ **Lead time, not detection** | We measure *how many steps before the SLA breach* we raise the flag. That's the whole point — time to intervene. |
| 🔒 **Air-gapped & grounded** | Local retrieval over runbooks; every recommendation is a **structured JSON card with a citation** using a **cite-or-abstain** rule (never fabricates a source). |
| 🧩 **Interpretable now, upgradeable later** | Transparent baseline+trend risk model today; swaps to an LSTM / Temporal-CNN forecaster + graph cascade layer behind the **same interfaces**. |
| 📊 **Measured, not hand-wavy** | Precision / recall / false-positive-rate **and lead time**, per scenario, shown on-screen. |
| 🖥️ **It actually runs** | A real Streamlit operator console with animated playback — not a mockup. |

---

## 🚀 Features

- **Risk timeline** — forecasts the breach and marks the flag, the breach, and the lead-time window.
- **Network topology** — the degrading MPLS/IPSec path glows green → red as risk climbs.
- **KPI tiles + risk gauge** — health, status, **lead time**, time-to-impact, confidence.
- **Grounded copilot card** — predicted issue, root cause, recommended actions, **citation**, cite-or-abstain.
- **Natural-language chat** — “why is this segment at risk?”, “what should I do?” — answers stay grounded.
- **Evaluation panel** — precision / recall / FPR per scenario + a **reactive-vs-predictive** comparison.
- **Trust & air-gap panel** — retrieved-context (RAG) transparency, a **0-egress monitor**, incident log, and a downloadable incident report.
- **Four fault scenarios** with animated time playback.

---

## 🖼️ Screenshots

<p align="center">
  <img src="deck/assets/dashboard_hero.png" width="90%" alt="Operator dashboard"><br>
  <sub><b>Operator console</b> — KPIs, at-risk topology, risk gauge, and forecast timeline.</sub>
</p>

<p align="center">
  <img src="deck/assets/dashboard_copilot.png" width="90%" alt="Grounded copilot"><br>
  <sub><b>Grounded copilot</b> — cited remediation card + natural-language chat, network-wide predictions.</sub>
</p>

---

## 🏗️ Architecture

<p align="center">
  <img src="deck/assets/architecture.png" width="92%" alt="Architecture">
</p>

Everything runs **inside the air-gap boundary — zero outbound traffic**:

`Network sim → Telemetry → Predictive engine → Offline copilot (LLM + local RAG) → Operator UI`

---

## ⚙️ How it works

### Prediction
1. **Learn** a healthy baseline from an initial calibration window.
2. **Score risk** = how far the leading indicator has travelled toward *critical*, **gated by a
   confirmed upward trend** (so a stable-but-high link doesn't false-alarm).
3. **Debounce** — require a sustained crossing → no flapping alerts.
4. **Project time-to-impact** from the leading indicator's trajectory.

<p align="center">
  <img src="deck/assets/precursor_signature.png" width="80%" alt="Precursor signature"><br>
  <sub>Precursors (utilization, jitter) trend up <b>before</b> the SLA metric breaches — that's what makes lead time possible.</sub>
</p>

### The copilot (anti-hallucination by design)
Prediction fires → **retrieve local runbook context** → emit a **structured JSON card**
(issue · confidence · root cause · affected sites · time-to-impact · recommended actions · **citation**).
If nothing is retrieved, it **abstains** rather than inventing a source.

> **Honest status:** the demo copilot is **template-backed** over a local runbook KB (no model
> download, runs anywhere). It's built behind a `generate_card()` interface so a **local quantized
> LLM (Ollama)** drops in for the full build *without changing the schema*.

---

## 📈 Results (reproducible, seed-fixed)

<p align="center">
  <img src="deck/assets/reactive_vs_predictive.png" width="80%" alt="Reactive vs predictive">
</p>

| Scenario | Leading signal | Lead time | Precision | FPR |
|---|---|:--:|:--:|:--:|
| Progressive hub-spoke congestion | link utilization | **21 steps** | 1.0 | 0.0 |
| BGP flap → reroute cascade | BGP flap rate | **27 steps** | 1.0 | 0.0 |
| Intermittent MPLS underlay loss | tunnel jitter | **17 steps** | 1.0 | 0.0 |
| Controller misconfig → policy drift | link utilization | **23 steps** | 1.0 | 0.0 |

Recall is **0.37–0.56** *by design* — the earliest fault steps carry no signal yet; what matters is
**early, false-alarm-free** detection (precision 1.0, FPR 0.0). Confidence climbs from ~75% at the
flag to ~99% at the breach.

---

## 🛠️ Tech stack

**Built now (this offline slice):** Python · NumPy · pandas · scikit-learn-free (pure numpy/pandas
model) · Streamlit · Plotly · Matplotlib. Runs on a modest laptop, **no network calls**.

**Full build / roadmap (air-gapped):** Containerlab + FRRouting (MPLS/LDP/SR, BGP, OSPF, VRFs, IPSec)
· Telegraf + InfluxDB/Prometheus + syslog/NetFlow · PyTorch (LSTM / Temporal-CNN) + graph layer ·
Ollama + a quantized 7–8B model (Phi-3-mini / Qwen2.5-7B / Llama-3.1-8B) · FAISS/Chroma + bge/MiniLM
embeddings.

---

## ⚡ Getting started

> Prerequisites: **Python 3.10+**. Everything installs from PyPI; the demo needs no GPU and no network at runtime.

```bash
# from the repository root
pip install -r requirements.txt

# 1) Interactive dashboard (the thing to demo / record)
streamlit run src/app.py         # opens http://localhost:8501

# 2) Headless run — writes chart + card + csv, prints the lead-time headline
python src/run_demo.py
```

Headless outputs land in `outputs/`: `telemetry.csv`, `risk_timeline.png`, `copilot_card.json`.

Example headline:

```
Risk flag at step 86 · SLA breach at step 107 · LEAD TIME = 21 steps
predicted time-to-impact 23 steps · card cited to RB-CONGESTION-01
```

---

## 📂 Project structure

```
netcopilot-demo/
├── src/
│   ├── generate_telemetry.py   # scenarios, topology, gradual precursor faults + ground truth
│   ├── predict.py              # baseline+trend risk model, lead time, time-to-impact, evaluate()
│   ├── copilot.py              # grounded/cited card, cite-or-abstain, NL chat, incident report
│   ├── run_demo.py             # headless CLI
│   └── app.py                  # Streamlit operator dashboard
├── data/runbooks.json          # local RAG corpus (the grounding knowledge base)
├── outputs/                    # generated artifacts (chart, card, csv)
├── deck/
│   ├── make_assets.py          # regenerate all deck visuals
│   ├── build_deck.py           # build the submission PPTX on the official template
│   ├── slide_specs.json        # slide copy
│   └── assets/                 # diagrams + dashboard screenshots
├── requirements.txt
└── README.md
```

---

## 🔒 Offline / air-gap guarantee

The demo makes **zero network calls at runtime** — model logic and runbook retrieval are entirely
local. The full build preserves this (local LLM + local vector index) and proves it with a packet
monitor showing **zero outbound connections** during a full predict → explain cycle.

---

## 🗺️ Roadmap

- [x] Synthetic telemetry with gradual precursor faults + ground-truth labels
- [x] Interpretable baseline+trend predictor with **lead time** + time-to-impact
- [x] Grounded, cited copilot (cite-or-abstain) + NL chat
- [x] Streamlit operator console (topology, gauge, timeline, evaluation, air-gap monitor)
- [x] Four fault scenarios + evaluation metrics
- [ ] Real emulated network (Containerlab + FRRouting) → live telemetry
- [ ] LSTM / Temporal-CNN forecaster + graph cascade propagation
- [ ] Local LLM (Ollama) + FAISS RAG behind the existing copilot interface
- [ ] Packet-monitor air-gap proof

---

## 👥 Team

**Team:** TechXCoders · **College:** Keshav Memorial Institute of Technology
**Team Leader:** Vishnu Priya Taduka
Built for **ISRO Bhartiya Antariksh Hackathon (BAH) 2026**.
