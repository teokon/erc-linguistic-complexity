# Linguistic Complexity as an Explainability Lens for ERC


---

## What We Show

We measure how linguistic complexity of utterances explains when and why conversational context helps EmoBERTa in Emotion Recognition in Conversation (ERC).

**Main finding:** Context helps most for linguistically *simple* utterances — short, ambiguous expressions that lack sufficient emotional signal on their own. For complex utterances, context is largely redundant.

**Mechanistic chain:**

```
Low LIX / MLT
    → High VADER ambiguity (0.820)
    → High single-utt model entropy (0.961)
    → Context reduces entropy by 0.553
    → Large F1 gain (+0.210)

High LIX / MLT
    → Low VADER ambiguity (0.641)
    → Lower single-utt entropy (0.772)
    → Context reduces entropy by 0.361
    → Small F1 gain (+0.047)
```

**Key results (IEMOCAP, Pearson r vs context gain):**

| Metric | r | sig |
|---|---|---|
| entropy_reduction | +0.229 | *** |
| LIX | −0.164 | *** |
| MLT (L2SCA) | −0.144 | *** |
| MLS (L2SCA) | −0.126 | *** |
| VADER ambiguity | +0.091 | *** |
| DC/C (L2SCA) | −0.041 | n.s. |

Partial correlation controlling for utterance length: LIX r=−0.087*** — the effect persists beyond length alone.

---

## Metrics

### Surface Metrics

**LIX** — Björnsson (1968)
Measures % of long words (>6 chars) + mean sentence length.
Formula: `W/S + 100 × LW/W`
- W = words, S = sentences, LW = words with >6 letters
- Strongest predictor (r=−0.164***)
- Captures both sentence length and lexical difficulty

**ARI** — Senter & Smith (1967)
Readability grade level based on characters per word and words per sentence.
Formula: `4.71 × C/W + 0.5 × W/S − 21.43`
- C = characters, W = words, S = sentences
- Weaker signal (r=−0.071**), flat until High bin

**MATTR** — Covington & McFall (2010)
Lexical diversity using a sliding window TTR, length-unbiased.
Formula: `avg(|V_i| / 50)` over windows of 50 tokens
- Cannot bin reliably on short utterances (many duplicates)

---

### Syntactic Metrics (L2SCA — computed locally with NeoSCA)

**MLT** — Wolfe-Quintero et al. (1998)
Mean Length of T-unit (main clause + dependent clauses).
Formula: `W ÷ T` where T = number of T-units
- Strongest syntactic metric (r=−0.144***)
- Captures production length, not subordination depth

**MLS** — Lu (2010)
Mean Length of Sentence.
Formula: `W ÷ S`
- Very similar to MLT (r=−0.126***)

**DC/C** — Lu (2010)
Dependent clauses per clause — measures subordination depth.
Formula: `DC ÷ C`
- n.s. (r=−0.041) — subordination does NOT drive the finding
- 75% zeros in short spoken utterances, cannot bin reliably

**ADD proxy** — Liu (2008)
Mean distance between consecutive content words (stop words removed).
Approximates Average Dependency Distance without a parser.
Formula: `mean(pos(w_{i+1}) − pos(w_i))` over content words
- Moderate signal (r=−0.124***)
- Non-monotonic pattern due to proxy approximation

---

### Semantic Metrics

**VADER ambiguity** — Hutto & Gilbert (2014)
Sentiment ambiguity: `1 − |compound score|`.
Compound score close to 0 = ambiguous sentiment.
- Positive correlation with gain (r=+0.091***)
- Monotonically decreases Low→High (0.820→0.641), matching gain pattern
- Confirms: simple utterances are sentimentally ambiguous → need context

**entropy_reduction** — Shannon (1948)
`H(single-utt model) − H(EmoBERTa)` where H = Shannon entropy of softmax.
- Strongest overall predictor (r=+0.229***)
- Single-utt entropy: Low=0.961, High=0.772 (model uncertain for simple utterances)
- EmoBERTa entropy: flat ~0.40 (context keeps it confident regardless)

**NRC emotion density** — Mohammad & Turney (2013)
Proportion of emotion-bearing words (8 basic emotions).
- n.s. (r=−0.013) — IEMOCAP is spoken dialogue: emotion is in context, not words
- Negative finding itself: lexicon-based methods fail on conversational data

---

## Repository Structure

```
erc-linguistic-complexity/
├── notebooks/
│   ├── rq3_clean.ipynb       # Main pipeline: inference + analysis (IEMOCAP)
│   └── pilot_v3_clean.ipynb  # RQ1 + RQ2 analysis (MELD)
├── scripts/
│   ├── l2sca_analysis.py     # Per-metric bin analysis (run locally after L2SCA)
│   └── rq2_stats.py          # RQ2 Mann-Whitney statistical tests
├── results/                  # Output plots (PNG)
├── requirements.txt
└── README.md
```

---

## How to Run

### Step 1 — Local: L2SCA syntactic metrics

> Requires Java 8+ and Python 3.11+. Run on Windows/Mac/Linux.

```bash
# 1. Install Java: https://www.java.com/en/download/
java -version   # verify

# 2. Install NeoSCA
pip install neosca

# 3. Download Stanford Parser (~500MB, first run only)
nsca --check-depends

# 4. Write utterances to .txt files (one per file)
#    from your rq3_results.csv:
python -c "
import pandas as pd, os
df = pd.read_csv('rq3_results.csv')
os.makedirs('l2sca_inputs', exist_ok=True)
for i, row in df.iterrows():
    with open(f'l2sca_inputs/utt_{i}.txt', 'w') as f:
        f.write(str(row['utterance_text']))
print(len(df), 'files written')
"

# 5. Run L2SCA (~15-30 min for 1600 utterances)
nsca l2sca_inputs\*.txt -o l2sca_results.csv     # Windows
nsca l2sca_inputs/*.txt -o l2sca_results.csv      # Mac/Linux

# 6. Merge with inference results and run analysis
python scripts/l2sca_analysis.py    # requires rq3_results_with_l2sca.csv
python scripts/rq2_stats.py         # RQ2 Mann-Whitney tests
```

---

### Step 2 — Google Colab: inference + full analysis

> Runs on Colab Free (T4 GPU). Estimated total time: ~20 min.

**1. Open the notebook**

Go to [colab.research.google.com](https://colab.research.google.com) → File → Upload notebook
→ select `notebooks/rq3_clean.ipynb`

**2. Set your paths in Cell 5**

```python
CHECKPOINT_CTX = '/content/drive/MyDrive/YOUR_EMOBERTA_CHECKPOINT'
CHECKPOINT_SNG = '/content/drive/MyDrive/YOUR_SINGLE_UTT_CHECKPOINT'
CONTEXT_CSV    = '/content/drive/MyDrive/test_constructed_targetSEPonly.csv'
SINGLE_CSV     = '/content/drive/MyDrive/iemocap6_emoberta_test.csv'
L2SCA_CSV      = '/content/drive/MyDrive/rq3_results_with_l2sca.csv'
OUTPUT_DIR     = '/content/drive/MyDrive/rq3_results_iemocap'
```

**3. Run all cells in order**

| Cells | What it does | Time |
|---|---|---|
| 1–4 | Install + imports + mount Drive + paths | ~2 min |
| 5 | Load both models | ~1 min |
| 6–8 | Load datasets + tokenize + sampling | ~3 min |
| 9–11 | Surface + L2SCA + semantic metrics | ~2 min |
| 12 | EmoBERTa inference (saves entropy) | ~5 min |
| 13 | Single-utt inference (saves entropy) | ~3 min |
| 14–18 | RQ3 analysis + correlation + partial corr | ~1 min |
| 19–20 | Plots + save to Drive | ~1 min |

**4. Outputs saved to Drive**

```
rq3_full_results.csv      utterance-level results with all metrics
rq3_summary.csv           F1 per complexity bin
correlation_full.csv      all metrics vs gain
rq3_main.png              grouped bar + line chart
correlation_all.png       all metrics correlation bar chart
```

---
## Installation

```bash
git clone https://github.com/YOUR_USERNAME/erc-linguistic-complexity.git
cd erc-linguistic-complexity
pip install -r requirements.txt
```

---



