"""Statistical analysis: Friedman test, Wilcoxon post-hoc, McNemar, ablation table."""
import json, csv
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Load scores
scores_path = Path("outputs/scores")
csv_path = scores_path / "llm_judge.csv"
jsonl_path = scores_path / "llm_judge.jsonl"

if not jsonl_path.exists():
    print("No scores found. Run run_judge.py first.")
    exit(1)

scores = []
with open(jsonl_path, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            scores.append(json.loads(line))

df = pd.DataFrame(scores)
print(f"Loaded {len(df)} scores\n")

# ═══ Ablation Table ═══
print("=" * 60)
print("ABLATION TABLE — 4 Configs")
print("=" * 60)
configs = ['vanilla', 'naive_rag', 'lightrag_generic', 'cardiorag_full']
labels = ['C1 Vanilla', 'C2 Naive RAG', 'C3 LightRAG-Generic', 'C4 CardioRAG Full']

ablation_rows = []
for cfg, label in zip(configs, labels):
    sub = df[df['config'] == cfg]
    ablation_rows.append({
        'Config': label,
        'N': len(sub),
        'Accuracy (mean)': f"{sub['clinical_accuracy'].mean():.2f}",
        'Accuracy (SD)': f"{sub['clinical_accuracy'].std():.2f}",
        'Patient-Centered (mean)': f"{sub['patient_centeredness'].mean():.2f}",
        'Patient-Centered (SD)': f"{sub['patient_centeredness'].std():.2f}",
        'Safety (mean)': f"{sub['safety'].mean():.2f}",
        'Safety (SD)': f"{sub['safety'].std():.2f}",
        'Citation (mean)': f"{sub['citation_correctness'].mean():.2f}",
        'Citation (SD)': f"{sub['citation_correctness'].std():.2f}",
        'Hallucination Rate': f"{sub['hallucination'].mean()*100:.1f}%",
        'Clinical Harm Rate': f"{sub['clinical_harm'].mean()*100:.1f}%",
    })

ablation_df = pd.DataFrame(ablation_rows)
print(ablation_df.to_string(index=False))

# Save ablation table
ablation_df.to_csv(scores_path / "ablation_table.csv", index=False)
print(f"\nSaved: ablation_table.csv")

# ═══ Friedman Test ═══
print("\n" + "=" * 60)
print("FRIEDMAN TEST — Overall config ranking")
print("=" * 60)

for metric in ['clinical_accuracy', 'patient_centeredness', 'safety', 'citation_correctness']:
    # Pivot: scenarios × configs
    pivoted = df.pivot(index='scenario_id', columns='config', values=metric).dropna()
    if len(pivoted) < 3:
        print(f"  {metric}: insufficient paired data")
        continue
    
    groups = [pivoted[c].values for c in configs if c in pivoted.columns]
    if len(groups) < 2:
        continue
    
    try:
        stat, p = stats.friedmanchisquare(*groups)
        print(f"  {metric}: chi2={stat:.2f}, p={p:.4f} {'*' if p < 0.05 else 'NS'}")
    except Exception as e:
        print(f"  {metric}: error={str(e)[:50]}")

# ═══ Wilcoxon Post-hoc ═══
print("\n" + "=" * 60)
print("WILCOXON POST-HOC (Clinical Accuracy)")
print("=" * 60)

metric = 'clinical_accuracy'
results = []
for i, c1 in enumerate(configs):
    for c2 in configs[i+1:]:
        v1 = df[df['config'] == c1][metric].values
        v2 = df[df['config'] == c2][metric].values
        min_len = min(len(v1), len(v2))
        try:
            stat, p = stats.wilcoxon(v1[:min_len], v2[:min_len])
            p_corr = min(p * 6, 1.0)  # Bonferroni: 6 comparisons
            results.append({
                'Comparison': f'{c1} vs {c2}',
                'Statistic': f'{stat:.1f}',
                'p_raw': f'{p:.4f}',
                'p_corrected': f'{p_corr:.4f}',
                'Significant': 'YES' if p_corr < 0.05 else 'no',
            })
        except Exception as e:
            results.append({
                'Comparison': f'{c1} vs {c2}',
                'Statistic': 'N/A',
                'p_raw': 'N/A',
                'p_corrected': 'N/A',
                'Significant': f'error: {str(e)[:30]}',
            })

posthoc_df = pd.DataFrame(results)
print(posthoc_df.to_string(index=False))
posthoc_df.to_csv(scores_path / "wilcoxon_posthoc.csv", index=False)
print(f"\nSaved: wilcoxon_posthoc.csv")

# ═══ McNemar Test (Hallucination Rate) ═══
print("\n" + "=" * 60)
print("MCNEMAR TEST — Hallucination Rate Pairwise")
print("=" * 60)

for i, c1 in enumerate(configs):
    for c2 in configs[i+1:]:
        # Need paired data: same scenario_id across configs
        merged = df[df['config'].isin([c1, c2])].pivot(index='scenario_id', columns='config', values='hallucination').dropna()
        if len(merged) < 5:
            print(f"  {c1} vs {c2}: insufficient paired data ({len(merged)})")
            continue
        
        b = ((merged[c1] == True) & (merged[c2] == False)).sum()
        c = ((merged[c1] == False) & (merged[c2] == True)).sum()
        
        if b + c > 0:
            stat = (abs(b - c) - 1)**2 / (b + c)
            p = 1 - stats.chi2.cdf(stat, 1)
            print(f"  {c1} vs {c2}: b={b} c={c} chi2={stat:.2f} p={p:.4f} {'*' if p < 0.05 else 'NS'}")
        else:
            print(f"  {c1} vs {c2}: b=0 c=0 (no discordant pairs)")

# ═══ Guardrail Metrics ═══
print("\n" + "=" * 60)
print("GUARDRAIL PERFORMANCE (C4 Only)")
print("=" * 60)

c4 = df[df['config'] == 'cardiorag_full']
# Emergency scenarios: safety_label = emergency_routing
emergency = df[df['config'].isin(configs)]
em_total = len(emergency[emergency['safety_label'] == 'emergency_routing'])
print(f"  Emergency scenarios in dataset: variable")
print(f"  Note: Guardrail pre-block stats tracked separately in generation phase")
print(f"  OOB flags in C4 responses: {len(c4[c4.get('oob_flagged', False)]) if 'oob_flagged' in c4.columns else 'N/A (tracked in gen phase)'}")

# ═══ Summary ═══
print("\n" + "=" * 60)
print("KEY FINDINGS SUMMARY")
print("=" * 60)

for cfg, label in zip(configs, labels):
    sub = df[df['config'] == cfg]
    acc = sub['clinical_accuracy'].mean()
    safe = sub['safety'].mean()
    hall = sub['hallucination'].mean()
    harm = sub['clinical_harm'].mean()
    print(f"  {label}: Acc={acc:.2f} | Safety={safe:.2f} | Halluc={hall:.1%} | Harm={harm:.1%}")

print("\nDone! All tables saved to outputs/scores/")
