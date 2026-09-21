# Data

All cases are **standardized virtual scenarios**. There are no real patient records, clinic transcripts, or identifiable demographics beyond synthetic fields such as age, sex, and LVEF in the scenario JSONL.

| Path | Contents |
|---|---|
| `cardiokg/*.jsonl` | Curated CardioKG nodes and relations for CAD, HF, and AF (401 node rows / 501 relation rows) |
| `scenarios/internal.jsonl` | 96-scenario development split |
| `scenarios/external.jsonl` | 24-scenario held-out synthetic split (same construction process; not a clinical external-validation cohort) |
| `figure_source/*.csv` | Source tables for the manuscript figures; filenames keep internal figure IDs, mapped to manuscript numbering in the top-level README |

Society guideline PDFs are copyrighted and are **not** included. Rebuild CardioKG from locally obtained guideline files if you need to regenerate the graph.
