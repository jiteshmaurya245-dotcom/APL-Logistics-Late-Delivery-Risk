# Project Brief

**Client:** APL Logistics (KWE Group)
**Program:** Unified Mentor internship

## Problem

APL Logistics' supply chain operations team has no way to flag which orders
are likely to arrive late *before* they ship. Delays are currently handled
reactively — after the fact — which is costly and erodes customer trust.

## Goal

Build a model and dashboard that scores every order's probability of a late
delivery, groups orders into Low / Medium / High risk tiers, and surfaces the
key factors driving that risk, so operations staff can intervene early
(reroute, expedite, or proactively contact the customer).

## Dataset

~180K historical orders with shipping, customer, product, and financial
fields. Full column dictionary is in the top-level `README.md`.

Two columns are intentionally excluded from modeling:
- `Days for shipping (real)` — the actual outcome the target is derived
  from; including it would leak the answer.
- `Delivery Status` — directly encodes lateness (e.g. "Late delivery",
  "Shipping on time"); same leakage problem.

## Methodology

1. **Preprocessing** — drop leakage/PII columns, handle missing values,
   one-hot encode categoricals, scale numeric features.
2. **Feature engineering** — shipping pressure index, discount/profit
   flags, tight-schedule flag.
3. **Modeling** — Random Forest classifier with class weighting to handle
   the imbalance between on-time and late orders.
4. **Evaluation** — ROC-AUC, precision, recall, F1, confusion matrix.
5. **Delivery** — Streamlit dashboard with live filters (shipping mode,
   market, segment, risk threshold), an order-level "what-if" predictor,
   and a region/mode risk breakdown.

## Deliverables

- Research paper (EDA, insights, recommendations)
- Streamlit dashboard (live analytics)
- Executive summary for government/stakeholder review
