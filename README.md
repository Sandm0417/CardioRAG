# CardioRAG

Guideline-grounded retrieval-augmented generation with a safety guardrail for cardiovascular patient communication.

This public package contains research code, synthetic virtual scenarios, CardioKG JSONL, and figure source tables. It does **not** contain API keys, `.env` files, real patient data, or copyrighted guideline PDFs.

## What is included

- `src/cardiorag/` — guardrail, input normalization, schemas, LightRAG adapter
- `app/streamlit_app.py` — local three-tab prototype (Overview / Screening / Chat)
- `scripts/` — v4 evaluation, heuristic scoring, optional LLM judging, figure rendering
- `data/cardiokg/` — 401 entity rows and 501 relation rows
- `data/scenarios/` — 96 development + 24 held-out synthetic scenarios
- `data/figure_source/` — CSV tables for the manuscript figures and the graphical abstract
- `tests/` — guardrail and normalization tests that do not need network access

## What is not included

- `.env` or live keys
- LightRAG vector indexes
- ESC / AHA guideline PDFs and slide decks
- Raw model generation logs

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[ui]"     # Streamlit demo
pip install -e ".[llm]"    # DeepSeek / OpenAI clients
pip install -e ".[rag]"    # LightRAG index rebuild
```

Copy `.env.example` to `.env` **only on your machine** if you need generation or index rebuild:

```bash
cp .env.example .env
```

## Tests (no API)

```bash
python -m pytest tests/test_guardrail.py tests/test_normalize.py -q
```

## Local prototype

Bind to localhost only:

```bash
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501). Chat uses `DEEPSEEK_API_KEY` from a local `.env` if present; Screening and Overview work without a key.

## Reproducing the paper analysis

Heuristic scoring and statistics do not require an API once generation files exist:

```bash
python scripts/fast_judge.py --input outputs/raw_generations/deepseek_full_zh.jsonl
python scripts/recompute_v4_primary.py
```

Redraw the manuscript figures and the graphical abstract from the bundled CSVs:

```bash
python scripts/render_figures.py              # Figures 1-3
python scripts/render_graphical_abstract.py   # graphical abstract
```

Outputs write to `outputs/figures/`.

| Manuscript figure | Script output | Source CSV |
|---|---|---|
| Figure 1, primary rubric scores | `figure1_primary_rubric.*` | `figure4_ablation.csv`, `figure3_metrics_heatmap.csv` |
| Figure 2, disease-stratified scores | `figure2_disease_robustness.*` | `figure6_disease_stratified.csv`, `figure7_paired_differences.csv` |
| Figure 3, routing and safety audit | `figure3_routing_safety.*` | `figure2_performance_by_dataset.csv`, `figure5_safety.csv` |
| Graphical abstract | `graphical_abstract.*` | inline values reported in the manuscript |

The CSV filenames keep their internal identifiers; the table above maps them to the manuscript figure numbers. The manuscript Figure 4 is a composite of screenshots of the local Streamlit app, so it is reproduced by running the prototype rather than by a plotting script. `figure1_kg_stats.csv` holds the CardioKG layer counts reported in supplementary Table S3.

Re-running generation (`scripts/run_eval_v4.py`) or LLM judging requires your own DeepSeek key and is optional.

## Data note

The 24-scenario split is a held-out **synthetic** split built by the same process as the development set. It is not a clinical external-validation cohort.

## License

MIT. Guideline text remains the property of the issuing societies and must be obtained by the user.
