# AutoAnalyst Finance

**A multi-agent LLM system for automated equity research and investment report generation.**

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://langchain-ai.github.io/langgraph/"><img src="https://img.shields.io/badge/LangGraph-orchestration-1C3C3C?logo=langchain&logoColor=white" alt="LangGraph"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-REST_API-009688?logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-inference-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://huggingface.co/ProsusAI/finbert"><img src="https://img.shields.io/badge/FinBERT-sentiment-FF9E0D" alt="FinBERT"></a>
  <a href="https://huggingface.co/docs/transformers"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Transformers-NLP-FFD21E?logo=huggingface&logoColor=black" alt="Transformers"></a>
  <a href="https://weasyprint.org/"><img src="https://img.shields.io/badge/WeasyPrint-PDF-111111" alt="WeasyPrint"></a>
  <a href="https://github.com/ranaroussi/yfinance"><img src="https://img.shields.io/badge/yfinance-market_data-5F0F40?logo=yahoo&logoColor=white" alt="yfinance"></a>
  <a href="https://fred.stlouisfed.org/"><img src="https://img.shields.io/badge/FRED-macro_data-2F6FAC" alt="FRED"></a>
  <a href="https://tavily.com/"><img src="https://img.shields.io/badge/Tavily-news_search-34A853" alt="Tavily"></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/Docker-containerized-2496ED?logo=docker&logoColor=white" alt="Docker"></a>
</p>

<p align="center">
  <a href="https://mazen-ben-brahim--autoanalyst-finance-frontend.modal.run"><img src="https://img.shields.io/badge/Live-Modal-7F00FF" alt="Live on Modal"></a>
  &nbsp;<strong>Try it:</strong>
  <a href="https://mazen-ben-brahim--autoanalyst-finance-frontend.modal.run">Streamlit UI</a>
  · <a href="https://mazen-ben-brahim--autoanalyst-finance-api.modal.run/docs">API docs</a>
</p>

AutoAnalyst Finance orchestrates a team of specialized agents — built on **LangGraph** — that research any publicly traded company and produce a structured, institutional-grade investment report (executive summary, fundamentals, valuation, DCF, sentiment, and risk) as a styled PDF in under two minutes. It pairs large language model (LLM) reasoning with grounded financial data (yfinance, FRED), deterministic valuation models, a local **FinBERT** sentiment classifier, and an auditable rule-based risk engine.



---

## Overview

Equity research is data-intensive: an analyst gathers price and financial data, computes and interprets ratios, builds a valuation model, gauges market sentiment, assesses risk, and finally synthesizes everything into a coherent report. AutoAnalyst Finance automates this end-to-end workflow as a pipeline of cooperating agents, each owning one stage:

| Stage | Agent | Output |
|---|---|---|
| Data collection | `data_collector` | price history, quarterly statements, macro indicators, peers |
| Fundamentals | `fundamental_analyst` | valuation/profitability/liquidity ratios, two-stage DCF, peers |
| Sentiment | `sentiment_analyst` | news + FinBERT scoring, aggregate sentiment, key topics |
| Risk | `risk_detector` | deterministic red-flag rules → 0–100 risk score |
| Synthesis | `report_writer` | LLM executive summary + PDF report |

The agents share a single `AnalysisState` and are coordinated by a LangGraph state machine with conditional routing. State is checkpointed (`MemorySaver`), and the REST API streams the graph node-by-node so progress is observable in real time.

---

## System Architecture

A sequential graph with a conditional branch that injects a critical-risk warning when the risk score exceeds 75/100.

```
┌─────────────────┐
│ data_collector  │   yfinance + FRED
└────────┬────────┘
         ▼
┌─────────────────────┐
│ fundamental_analyst │   ratios (TTM ROE/ROA) · two-stage DCF · peers
└────────┬────────────┘
         ▼
┌────────────────────┐
│ sentiment_analyst  │   Tavily news + FinBERT
└────────┬───────────┘
         ▼
┌─────────────────┐      risk_score > 75 ?
│  risk_detector  │ ───────────────────► ┌──────────────────────┐
└────────┬────────┘                      │ add_critical_warning │
         │                               └───────────┬──────────┘
         ▼                                           ▼
              ┌─────────────────┐
              │  report_writer  │   LLM synthesis + WeasyPrint
              └────────┬────────┘
                       ▼
                    PDF report
```

| Agent | Responsibility | Tools / Models |
|---|---|---|
| `data_collector` | Price, statements, macro, peers | yfinance, FRED |
| `fundamental_analyst` | Ratios, DCF, peer comparison, interpretation | LLM |
| `sentiment_analyst` | News search, sentiment scoring, key topics | Tavily, FinBERT |
| `risk_detector` | Deterministic red-flag rules + risk score | rule engine |
| `report_writer` | Executive summary + full PDF report | LLM, WeasyPrint |
| `orchestrator` | Graph wiring + conditional routing | LangGraph |

---

## Methodology

### Fundamental analysis
Pulls quarterly income statements and balance sheets, then computes:
- **Valuation multiples** — P/E, forward P/E, P/B, P/S, EV/EBITDA, market cap.
- **Profitability** — net and EBITDA margins; **ROE/ROA** computed with **trailing-twelve-month (TTM) net income** so the flow matches the point-in-time balance-sheet stock (a single quarter would understate returns roughly four-fold).
- **Liquidity & leverage** — current ratio, debt-to-equity, interest coverage, free cash flow, FCF yield.

**Discounted cash flow (DCF).** A two-stage model: five years of explicit free-cash-flow projections followed by a Gordon-growth terminal value, discounted at a configurable WACC. The stage-one growth rate is a **blended estimate** of recent revenue and earnings growth, anchored to a long-run rate and capped at 25% — rather than feeding in a single, volatile quarterly figure.

**Peer comparison** aggregates sector-competitor multiples into sector-average P/E, P/B, and margin benchmarks. A final LLM pass writes a plain-language interpretation.

### Sentiment analysis
Recent news is retrieved via Tavily; each headline/snippet is classified by **FinBERT** (`ProsusAI/finbert`), a BERT model fine-tuned on financial text. Per-article positive/negative/neutral scores are aggregated into an overall label and a set of key topics.

### Risk detection
A transparent, **deterministic** rule engine — no LLM participates in scoring, so the result is reproducible and auditable. Each triggered rule contributes severity-weighted points; the total is capped at 100.

| Rule | Severity | Condition |
|---|---|---|
| High debt burden | high | debt/equity > 200% |
| Negative free cash flow | high | FCF < 0 |
| Low current ratio | high | current ratio < 1 |
| Poor interest coverage | high | interest coverage < 2× |
| Declining revenue | medium | revenue growth < −5% YoY |
| Compressed margins | medium | net margin < 3% |
| Negative ROE | medium | ROE < 0 |
| Negative news sentiment | medium | aggregate sentiment = bearish |
| Expensive valuation | low | P/E > 50 |

### Report synthesis
An LLM composes the executive summary and a full Markdown report from the shared state; WeasyPrint renders it to a styled PDF.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph, LangChain |
| LLM | Any OpenAI-compatible provider (Groq · OpenAI · Gemini · OpenRouter · Ollama) |
| Financial data | yfinance, FRED (fredapi) |
| News | Tavily |
| Sentiment | FinBERT via HuggingFace Transformers + PyTorch (CPU) |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit + Plotly |
| Reporting | WeasyPrint (PDF) |
| Deployment | Docker, Modal |

---

## Installation

### Prerequisites
- Python 3.11+
- An **LLM API key** (any OpenAI-compatible provider — see `.env.example`). Groq (free) is the default.
- A **Tavily** key (news). A **FRED** key is optional (macro; a fallback is used if absent).
- WeasyPrint system libraries (pango/cairo):
  - Ubuntu: `sudo apt-get install libpango-1.0-0 libpangoft2-1.0-0 libcairo2 libgdk-pixbuf2.0-0`
  - macOS: `brew install pango cairo`
  - Windows: [GTK3 runtime](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases)

### Steps
```bash
git clone <repo-url> && cd autoanalyst-finance
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then fill in your keys
```

### Choosing an LLM provider
`src/tools/llm.py` is provider-agnostic. Set three variables in `.env` (`LLM_*` take precedence over `OPENAI_API_KEY`):

| Provider | `LLM_BASE_URL` | `LLM_MODEL` |
|---|---|---|
| Groq (default, free) | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| OpenAI | *(uses `OPENAI_API_KEY`)* | `gpt-4o-mini` |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.0-flash` |
| OpenRouter | `https://openrouter.ai/api/v1` | `meta-llama/llama-3.3-70b-instruct:free` |
| Ollama (local) | `http://localhost:11434/v1` | `llama3.1` |

---

## Usage

### CLI
```bash
python scripts/run_analysis.py --ticker AAPL
python scripts/run_analysis.py --ticker AAPL --depth deep
python scripts/run_analysis.py --ticker BNP.PA --open     # open the PDF when done
```

### REST API
```bash
uvicorn src.api.main:app --reload
# POST /analyze            {"ticker": "AAPL", "analysis_depth": "quick"}
# GET  /status/{run_id}    incremental progress + completed steps
# GET  /report/{run_id}
# GET  /report/{run_id}/pdf
# GET  /health
```

### Streamlit frontend
```bash
streamlit run src/frontend/app.py            # http://localhost:8501
```

### Docker
```bash
docker compose up -d                                         # API :8000 · frontend :8501
API_PORT=18000 FRONTEND_PORT=18501 docker compose up -d     # alternate ports
docker compose down
```

### Deploy (Modal)
The full stack runs serverless on [Modal](https://modal.com/) — FastAPI and Streamlit
on a shared image with FinBERT baked in at build time (cold starts load it from disk
instead of re-downloading ~1.3 GB). See [`modal_app.py`](modal_app.py).

| Endpoint | URL |
|---|---|
| Streamlit UI | <https://mazen-ben-brahim--autoanalyst-finance-frontend.modal.run> |
| REST API (`/docs`) | <https://mazen-ben-brahim--autoanalyst-finance-api.modal.run> |

> Deployed via `modal deploy`; the CI workflow ([`deploy-modal.yml`](.github/workflows/deploy-modal.yml))
> auto-deploys on push to `main` once the `MODAL_TOKEN_*` repo secrets are set. The API
> scales to zero when idle (~10–20 s cold start).

```bash
pip install modal && modal token new
modal secret create autoanalyst-finance LLM_API_KEY=… TAVILY_API_KEY=… …
modal deploy modal_app.py       # prints the api + frontend URLs
```

---

## Sample Reports

Generated end-to-end by the pipeline. The `BNP.PA` row demonstrates support for non-US exchanges.

| Ticker | Company | Report |
|---|---|---|
| AAPL | Apple Inc. | [samples/AAPL.pdf](samples/AAPL.pdf) |
| MSFT | Microsoft Corp. | [samples/MSFT.pdf](samples/MSFT.pdf) |
| NVDA | NVIDIA Corp. | [samples/NVDA.pdf](samples/NVDA.pdf) |
| TSLA | Tesla Inc. | [samples/TSLA.pdf](samples/TSLA.pdf) |
| BNP.PA | BNP Paribas | [samples/BNP_PARIS.pdf](samples/BNP_PARIS.pdf) |

> Live reports are written to the gitignored `reports/` directory at runtime.

---

## Evaluation & Testing

The `pytest` suite is split into offline and network-marked tests:
```bash
pytest -m "not network"            # offline — no keys/internet required
pytest -m network                  # full live suite (LLM/yfinance/Tavily/FRED + PDF)
pytest -m network -k full_pipeline # end-to-end pipeline test
```
The network suite exercises the complete graph against live data sources and asserts that a PDF is produced.

---

## Project Structure

```
autoanalyst-finance/
├── src/
│   ├── agents/        # LangGraph nodes (data, fundamental, sentiment, risk, report)
│   ├── tools/         # yfinance, FRED, Tavily, FinBERT, WeasyPrint, LLM client
│   ├── state/         # AnalysisState TypedDict + factory
│   ├── prompts/       # LLM prompt templates
│   ├── api/           # FastAPI app + schemas
│   └── frontend/      # Streamlit app
├── scripts/run_analysis.py
├── tests/
├── samples/           # example PDF reports (tracked)
├── reports/           # runtime PDF output (gitignored)
├── modal_app.py       # serverless (Modal) deployment
├── docker-compose.yml · Dockerfile.api · Dockerfile.frontend
├── .github/workflows/ # CI + Modal deploy
├── requirements.txt · .env.example
└── README.md
```

---

## Limitations & Future Work

- **In-memory run store** — the API tracks runs per process, suitable for a single instance. For horizontal scaling, externalize run state (e.g., Redis).
- **DCF sensitivity** — the model uses configurable defaults (WACC, terminal growth); outputs are estimates sensitive to these assumptions.
- **Data provenance** — fundamentals rely on yfinance, which may lag or differ from primary regulatory filings.
- **Sentiment scope** — limited to recent news retrieved by Tavily; not a full social-media sentiment model.
- **Future work** — analyst-consensus growth estimates, multi-currency normalization, persistent report storage, and authentication.

---

## References

- **LangGraph** — agent orchestration. [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)
- **FinBERT** (`ProsusAI/finbert`) — financial sentiment classification. [huggingface.co/ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
- **yfinance** — market data. [github.com/ranaroussi/yfinance](https://github.com/ranaroussi/yfinance)
- **FRED** — Federal Reserve Economic Data. [fred.stlouisfed.org](https://fred.stlouisfed.org/)
- **Tavily** — LLM-oriented web search. [tavily.com](https://tavily.com/)
- **WeasyPrint** — HTML/CSS to PDF. [weasyprint.org](https://weasyprint.org/)

---


