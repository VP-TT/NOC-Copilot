# Offline Predictive Network Copilot — Round-One Demo

**Forecast, don't just detect.** A fully-offline vertical slice that predicts a network SLA
breach *before* it happens, measures the **lead time**, and emits a **grounded, cited**
remediation card — the core "predict → explain" loop of the full project.

This is intentionally a thin slice: no emulator, no model download, pure Python. But it is
**not throwaway** — the synthetic generator and the swappable copilot interface are the
day-one seed of the full build's ML and LLM tracks.

## What it does

1. **Generates** synthetic network telemetry with a *gradual* precursor fault
   (progressive hub-spoke congestion) plus a ground-truth SLA-breach label.
2. **Predicts** risk from the leading indicator (learned baseline + trend gate), raises a
   sustained "risk rising" flag, and computes **lead time = breach step − flag step**.
3. **Explains** the prediction as a structured JSON copilot card grounded in a local runbook,
   with a **citation** (no model, no network — template now, LLM-swappable later).

## Run it

```bash
pip install -r requirements.txt      # numpy, pandas, matplotlib  (offline-installable)
python src/run_demo.py               # from the project root
```

Outputs (written to `outputs/`):
- `telemetry.csv` — full risk-augmented time series
- `risk_timeline.png` — the money chart: risk flag clearly **before** the SLA breach
- `copilot_card.json` — structured, cited remediation card

Example headline (reproducible, seed-fixed):
> Risk flag at step 86 · SLA breach at step 107 · **LEAD TIME = 21 steps** ·
> predicted time-to-impact 23 steps · card cited to `RB-CONGESTION-01`.

## Recording the demo (60–90s)

1. Clear the terminal, run `python src/run_demo.py` — let the **LEAD TIME** line land.
2. Open `outputs/risk_timeline.png` — point at the green flag line vs the red breach line.
3. Open `outputs/copilot_card.json` — highlight `grounded: true` and the `citation`.
That clip + the chart are slides 10 of `PPT_OUTLINE.md`.

## Layout

```
netcopilot-demo/
  data/runbooks.json          # local grounding KB (the "RAG" corpus)
  src/generate_telemetry.py   # synthetic series + ramped precursor fault + ground truth
  src/predict.py              # baseline+trend risk model, lead-time + time-to-impact
  src/copilot.py              # generate_card(): structured, cited card (LLM-swappable)
  src/run_demo.py             # generate -> predict -> explain -> chart
  architecture.svg            # editable architecture diagram for the deck
  PPT_OUTLINE.md              # slide-by-slide proposal content
```

## How this grows into the full project

| Slice today | Full build (post-shortlist) |
|---|---|
| Synthetic telemetry | Containerlab + FRR emulator → real telemetry (same schema) |
| Baseline + trend detector | LSTM / Temporal-CNN forecaster + time-to-impact regressor + graph cascade layer |
| `generate_card()` template | Same interface backed by Ollama + quantized LLM + FAISS RAG |
| One scenario | All four required scenarios |
| Matplotlib chart | Streamlit operator UI (topology, alert queue, copilot chat) |

The interfaces (`telemetry → predict → card`) are stable, so each upgrade is a swap, not a rewrite.

## Offline / air-gap note

Round one runs with **zero network calls** — no model download, no API. The full build keeps
this property (local LLM + local vector index) and proves it with a packet monitor showing
zero outbound during a predict→explain cycle.
