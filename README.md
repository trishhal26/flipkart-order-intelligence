# Flipkart Order Intelligence & Support Assistant

A single connected system for Flipkart's catalog/support team: a return-risk model (Part 1),
a product-image categoriser (Part 2), and a LangGraph support agent (Part 3) that calls both
as real tools alongside a policy-question RAG pipeline.

## Repository Structure

```
flipkart-order-intelligence/
├── generate_orders.py                          # Part 1: seeded dataset generator
├── orders_dataset.csv                          # Part 1: generated dataset (6,000 rows)
├── notebooks/
│   ├── part1_return_risk_analysis.ipynb        # Part 1: EDA, preprocessing, baseline
│   ├── part1_return_risk_modeling.ipynb        # Part 1: LR/RF tuning, SHAP, subgroup analysis
│   ├── part2_product_image_categorizer.ipynb   # Part 2: transfer learning classifier
│   ├── part3_support_agent.ipynb               # Part 3: LangGraph agent
│   ├── models/
│   │   ├── return_risk_model.pkl               # Part 1 saved artifact
│   │   └── product_classifier.pt               # Part 2 saved artifact
│   └── data/
│       └── sample_images/                      # Part 2: 7 real exported test images
├── transcripts/                                # Part 3: 8 saved test conversation transcripts
└── README.md
```

## Setup

```bash
pip install numpy pandas scikit-learn joblib
pip install torch torchvision
pip install langgraph langchain-core sentence-transformers faiss-cpu
```

All dependencies are free and local — no API keys or paid accounts required anywhere in this project.

---

## Part 1 — Return-Risk Scoring Pipeline

**Regenerate the dataset (deterministic, seeded):**
```bash
python3 generate_orders.py
```
This produces `orders_dataset.csv` (6,000 rows, ~18-27% return rate).

**Run the analysis and modeling notebooks (in order):**
1. `notebooks/part1_return_risk_analysis.ipynb` — EDA, missingness analysis, preprocessing pipeline, baseline DummyClassifier, Logistic Regression + threshold sweep
2. `notebooks/part1_return_risk_modeling.ipynb` — Random Forest + GridSearchCV, feature importance (impurity + permutation), subgroup analysis, final artifact save

**Result:** Tuned Random Forest saved to `notebooks/models/return_risk_model.pkl`.
**F1-maximising threshold (t\*_rf): 0.47** — recall 0.5861, precision 0.3071, F1 0.4030.
This threshold anchors Part 3's `check_return_risk` risk buckets.
---

## Part 2 — Product Image Categoriser via Transfer Learning

**Run the notebook:**
notebooks/part2_product_image_categorizer.ipynb

- Dataset: Fashion-MNIST (pinned source: https://github.com/zalandoresearch/fashion-mnist), downloaded automatically via `torchvision.datasets.FashionMNIST`.
- Splits: 54,000 train / 6,000 validation (stratified) / 10,000 test (untouched until final evaluation).
- Backbone: pretrained **ResNet-18**, frozen, with a new classifier head (512 → 128 → 10) trained on cached backbone features (feature-extraction approach — see brief's speed tip).
- Hyperparameters: Optimizer **Adam**, learning rate **1e-3**, batch size **128**, **15 epochs**.
- **Feature-extraction validation accuracy: 90.52%** — well above the 80% bar, so fine-tuning (unfreezing backbone layers) was **not required**.
- **Final test-set accuracy: 89.94%**.
- Most-confused category pairs (from the real confusion matrix): **T-shirt/top ↔ Shirt** (210 confusions) and **Coat ↔ Shirt** (144 confusions) — both explained in the notebook by visual silhouette similarity at low resolution.

**Result:** Trained head saved to `notebooks/models/product_classifier.pt`. 7 real test-split images exported to `notebooks/data/sample_images/` for use by Part 3's `classify_product_image` tool.
---

## Part 3 — Flipkart Support Agent

**Run the notebook:**
notebooks/part3_support_agent.ipynb

- **Knowledge base:** 14 policy documents, chunked sentence-wise into 33 chunks, each mapped back to its parent document.
- **Retrieval:** Chunks embedded with `all-MiniLM-L6-v2` (free, local sentence-transformer), indexed with **Faiss** (`IndexFlatIP`, cosine similarity via L2-normalized vectors).
- **Tools (real, not hardcoded):**
  - `check_return_risk(order_features)` — loads Part 1's `return_risk_model.pkl`, buckets risk relative to **t\*_rf = 0.47** (Low < 0.47, High ≥ 0.62, Medium between).
  - `classify_product_image(image_path)` — loads Part 2's `product_classifier.pt`, run against real `.png` files in `data/sample_images/`.
- **Graph:** 5 nodes (`guardrail`, `intent`, `rag`, `tool`, `respond`) with conditional edges branching on both the injection-guardrail result and the classified intent.
- **State:** `conversation_history` and `last_order_id` are carried across turns within one conversation; a fresh `agent_graph.invoke()` call starts with both empty/`None` (see `transcripts/05_multiturn.txt` vs `transcripts/06_fresh_conversation.txt`).
- **Prompt engineering:** System prompt documented against the 4S principles (Specific, Short, Surround, Single) + role prompting, with 2 few-shot intent-classification examples — see notebook Task 7 for the full annotated prompt.
- **MOCK_LLM mode:** Default and only mode used for all graded transcripts. Zero API keys, zero network calls — the response generator deterministically composes structured `{answer, source, confidence}` JSON from retrieved chunks / tool outputs.
- **Guardrails:**
  - *Input-side:* prompt-injection pattern filter (blocks phrases like "ignore previous instructions", "pretend you are") — routes straight to a refusal, bypassing intent/RAG/tools entirely.
  - *Output-side:* groundedness check — refuses to answer a policy question if the top retrieved chunk's similarity is below **0.60**, printing the similarity score and threshold rather than fabricating an answer.

### Test Transcripts

All 8 required scenarios are saved in [`transcripts/`](./transcripts):

| File | Scenario |
|---|---|
| [`01_policy_apparel_return.txt`](./transcripts/01_policy_apparel_return.txt) | Policy question — apparel return window |
| [`02_policy_cod_refund.txt`](./transcripts/02_policy_cod_refund.txt) | Policy question — COD refund timeline |
| [`03_return_risk.txt`](./transcripts/03_return_risk.txt) | Return-risk question (real model call) |
| [`04_product_category.txt`](./transcripts/04_product_category.txt) | Product-category question (real image classifier call) |
| [`05_multiturn.txt`](./transcripts/05_multiturn.txt) | Multi-turn conversation — state carried across turns |
| [`06_fresh_conversation.txt`](./transcripts/06_fresh_conversation.txt) | Fresh conversation — state correctly absent/reset |
| [`07_prompt_injection.txt`](./transcripts/07_prompt_injection.txt) | Prompt-injection attempt — blocked by input-side guardrail |
| [`08_ungrounded_question.txt`](./transcripts/08_ungrounded_question.txt) | Ungrounded question — refused by output-side groundedness check |

### Example Transcript (Return-Risk Question)
============================================================
RETURN-RISK QUESTION

User input: Will order #7788 likely be returned?
Final answer: {'answer': 'This order has a predicted return probability of 0.6245 (High risk).', 'source': 'return_risk_tool', 'confidence': 0.6245}
Full state snapshot: intent=return_risk, blocked=False, last_order_id=7788

### Retrieval Evaluation (Precision@3 / Recall@3)

Evaluated across 6 realistic queries (document-level scoring, deduplicated):

| Query | Relevant Docs | Retrieved (top-3) | Precision@3 | Recall@3 |
|---|---|---|---|---|
| "How many days do I have to return a shirt I bought?" | doc_01 | doc_01, doc_03, doc_02 | 0.333 | 1.0 |
| "When will I get my refund if I paid cash on delivery?" | doc_04 | doc_04, doc_05, doc_11 | 0.333 | 1.0 |
| "My laptop arrived broken, what do I do?" | doc_10 | doc_10, doc_14, doc_02 | 0.333 | 1.0 |
| "Can I cancel my order after it has shipped?" | doc_11 | doc_11 | 1.0 | 1.0 |
| "How long does delivery usually take?" | doc_06, doc_07 | doc_06, doc_09, doc_07 | 0.667 | 1.0 |
| "Is pickup available for returns in my area?" | doc_08, doc_09 | doc_08, doc_09, doc_03 | 0.667 | 1.0 |

**Average Precision@3: 0.5555**
**Average Recall@3: 1.0** — every relevant document was retrieved within the top 3 for every query.
---

## Git Workflow

This repository's commit history includes a feature branch (`feature/part2-image-classifier`)
created, committed to twice, and merged into `main` via an explicit merge commit — visible via:
```bash
git log --oneline --graph --all
```

## Originality

The dataset-generation script's output, trained models, agent code, and written analysis in
this repository are original work produced for this capstone brief.





