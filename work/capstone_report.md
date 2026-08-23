# Capstone Report — Refresh / Content Opportunity Scoring
- **Author:** Raju Ahmad
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/raju-cse/Flyrank-internship_ml
- **Date:** August 2026

## 0. Abstract
This project asks: can we predict which FlyRank client pages will decline in search
visibility before it happens, so editors know where to focus refresh effort? Using 12
months of historical GSC/GA4 performance data plus content metadata (word count,
optimization history, ranking position) from the FlyRank warehouse, we trained a Random
Forest classifier to label pages as growing, stable, or declining based on a held-out
6-month future window. The model reached a macro-F1 of 0.43 versus a 0.30 majority-class
baseline, and correctly flagged 26% of truly declining pages (vs 0% for the baseline),
with word count, days-since-last-optimization, and past impressions as the strongest
signals. The output is a ranked, reason-coded action list that content editors can use to
prioritize which pages to refresh first.

## 1. Problem framing
**Unit of analysis:** individual content page (`content_hash_id`), scored monthly.
**Output:** a declining-probability score (0–1) per page, plus a reason code
(stale_content, thin_content, low_ranking_position, low_ctr).
**Action:** a content editor uses the ranked list to decide which pages to refresh,
rewrite, or leave alone this sprint, instead of reviewing pages randomly or by gut feel.
**Cost of a wrong call:** a false negative (missing a real decline) means a page keeps
losing visibility unnoticed; a false positive means wasted editor time on a page that was
fine. Because declining pages are the minority class, we prioritize recall on that class
over raw accuracy.
**Why ML helps:** with hundreds of thousands of pages and only a handful of editors,
manual review doesn't scale. A ranked, explainable score turns an unmanageable list into a
short, prioritized queue.

## 2. Data safety
**Data used:** `fact_content_daily_performance` (GSC/GA4 daily metrics, Jan 2025–Jun 2026)
and `dim_content` (content metadata: word count, creation/optimization dates, search
volume, competition). `dim_clients` was used only to confirm join keys, not as a feature.

**Excluded:** no client names, URLs, or raw queries were used anywhere — all joins use
pseudonymous `client_hash_id` / `content_hash_id`, which are used strictly for grouping
(train/test split) and never as model features.

**Leakage risks considered:**
- Features are built only from the past window (2025-01 to 2025-12); the label is derived
  only from the future window (2026-01 to 2026-06) — the two never overlap.
- Train/test split is grouped by `client_hash_id` (GroupShuffleSplit), so no client appears
  in both sets, preventing the model from learning client-specific shortcuts.
- Hash IDs are used only as join/group keys, never as features.

## 3. Baseline
The baseline is the majority-class rule: always predict "growing" (80.2% of the test set).
This is a fair, transparent comparison point — any model must beat what an editor gets by
doing nothing analytical at all.

Baseline results (test set, n=36,591):
- Accuracy: 80.2%
- Macro F1: 0.30
- Declining-class recall: 0.00 (by construction — it never predicts declining)

## 4. Model / analysis
**Method:** Random Forest classifier (200 trees, max_depth=12, min_samples_leaf=20,
class_weight='balanced' to counter class imbalance).

**Why it fits:** tabular, mixed-type features (counts, ratios, dates) with likely
non-linear interactions (e.g., word count matters differently depending on optimization
recency) — tree ensembles handle this without heavy preprocessing and give interpretable
feature importances.

**Features used (13):** impressions_past, clicks_past, avg_position_past, sessions_past,
engaged_sessions_past, ai_sessions_past, h1_h2_ratio (within-window trend), ctr_past,
content_age_days, days_since_optimized, word_count, search_volume, competition.

**Deliberately excluded:** raw hash IDs (join keys only), any post-cutoff (future window)
metric, and `report_date`/`month` directly (only used to build the aggregation windows).

**Label definition (one sentence):** a page is "declining" if its future-window
(2026-01–06) monthly-average impressions fell 20%+ vs. its past-window (2025)
monthly-average, "growing" if it rose 20%+, and "stable" otherwise; pages with zero past
impressions were excluded as insufficient data.

## 5. Evaluation
**Split:** grouped by `client_hash_id` (80/20, GroupShuffleSplit, seed=42) — never a random
row-level split — so no client's pages leak between train and test.

**Metrics (test set, n=36,591; base rate/majority class = 80.2% growing):**

| Model | Accuracy | Macro F1 | Declining Recall | Declining Precision |
|---|---|---|---|---|
| Baseline (majority) | 0.802 | 0.30 | 0.00 | — |
| Random Forest | 0.681 | 0.43 | 0.26 | 0.41 |

Raw accuracy alone is misleading here because of the 80% base rate — macro F1 and
per-class recall are the honest metrics for this imbalanced task.

**Error analysis:** the model's main failure mode is under-flagging true declines — 2,940
of 4,592 truly declining pages (64%) were missed and predicted "growing." This is the
model's main limitation and is called out explicitly in Section 6.

## 6. Interpretation
Feature importances (Random Forest, Gini-based):
1. word_count (22.0%)
2. days_since_optimized (21.3%)
3. impressions_past (20.5%)
4. content_age_days (10.2%)
5. avg_position_past (6.5%)

In plain words: pages that haven't been touched in a long time and are thin on content
are the pages most likely to decline — this is an intuitive, decision-support-friendly
result, not a surprising one. AI-referral sessions (`ai_sessions_past`) had almost no
predictive power (0.05%), likely because AI-referral traffic is still sparse across the
dataset — this is a negative result worth reporting rather than hiding.

We also observed that declining-labeled pages were on average ~20 days older than growing
pages (182 vs. 162 days) — a real but modest effect, consistent with content decay, though
age alone is a weak predictor compared to optimization recency and word count.

## 7. Recommendation
The model output is a ranked list of pages by `declining_probability`, each tagged with
reason codes (stale_content, thin_content, low_ranking_position, low_ctr). An editor would:
1. Pull the top N pages by declining_probability each sprint.
2. Read the reason code(s) to decide the fix: stale_content → refresh/update;
   thin_content → expand; low_ranking_position/low_ctr → review metadata and on-page intent
   match.
3. Re-run the scorer monthly to track whether refreshed pages move out of the high-risk
   list.

**Confidence:** directional, decision-support only — this ranks relative refresh priority;
it does not prove a page will decline, nor does it measure or claim any causal effect of
Google's algorithm. Given 64% recall on the declining class, this list should supplement,
not replace, editorial judgment — it's a triage tool, not a verdict.

## 8. Reproducibility
**Environment:** Google Colab, Python 3, key packages: duckdb, pandas, scikit-learn,
matplotlib, seaborn, huggingface_hub (see `requirements.txt`).

**Re-run steps:**
1. Clone the repo.
2. Open `work/notebooks/capstone.ipynb` in Colab.
3. Add your own Hugging Face read token as a Colab secret named `HF_TOKEN` (do not hardcode
   it — see `DATA_USE.md`).
4. Run all cells top to bottom. Random seed = 42 throughout (GroupShuffleSplit,
   RandomForestClassifier).
5. Outputs (confusion_matrix.png, feature_importance.png, baseline_vs_rf.png) are saved to
   `work/figures/`.

No sealed/holdout claim is made beyond the standard train/test split described in Section 5.

## 9. Acknowledgments & data credit
Built on the FlyRank ML Internship dataset — [flyrank.ai](https://flyrank.ai)
