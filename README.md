# Agricultural RAG & BERT Classification

**Language:** English · [Original (mixed TR/EN)](./README.original.md)

Text classification and retrieval-augmented generation (RAG) for the agricultural domain: BERT-family classifiers that route agricultural questions into six categories, plus a sentence-transformers retrieval pipeline, with Streamlit/API front ends and NVIDIA Jetson deployment scripts.

**Status:** Deployment-oriented research prototype. Classification results below are measured from committed training artifacts; RAG-quality and Jetson-latency numbers are **not yet benchmarked** (see [Limitations](#limitations)).

---

## Key results (measured)

| Model | Split | Accuracy | F1 | Source artifact |
|---|---|---:|---:|---|
| DistilBERT (best epoch) | Validation | 94.4% | 0.944 | `CreateModel/distilbert_agricultural/training_history.json` |
| DistilBERT (final epoch) | Validation | 93.7% | 0.937 | same |
| BERT-small (final) | Validation | 85.2% | 0.854 | `CreateModel/bert_small_agricultural/training_history.json` |

> These are validation-split results from the committed training histories. A locked-down test-set evaluation script and a fixed seed are still **TODO** (see roadmap). Any larger BERT-base/BERT-large numbers that appeared in earlier documentation are **not backed by artifacts in this repo** and have been removed.

## Data assets

The repository contains three *distinct* data collections. They are **not** additive — do not sum them into a single "dataset size".

### Classification dataset (used for the measured results above)

| Split | Rows | Per-category |
|---|---:|---|
| Train (`Data/train.csv`) | 1,262 | 210 × 6 categories (+2 `general_agriculture`) |
| Validation (`Data/val.csv`) | 270 | 45 × 6 categories |
| Test (`Data/test.csv`) | 271 | 45 × 6 categories (+1 `general_agriculture`) |

- **Categories (6):** `plant_disease`, `crop_management`, `plant_genetics`, `environmental_factors`, `food_security`, `technology`. A negligible 7th label (`general_agriculture`, 2–3 rows total) exists in the label map and should be dropped or folded in.
- **Format:** `text,label` CSV. **Label source:** category-labeled synthetic/curated agricultural text.
- **Split strategy:** stratified, balanced per category.

### Combined labeled pool

- `Data/agricultural_bert_dataset.csv` — **1,803 rows** (`text,label`). This is the fuller labeled pool from which the balanced train/val/test split above was drawn. `agricultural_bert_detailed.csv` is a richer-columns variant of the same content.

### Sentiment set (auxiliary, not used in the classification results)

- `Data/agricultural_sentiment.csv` — **780 rows**, a separate auxiliary set. Not part of the 6-category classification evaluation.

### Retrieval corpus (RAG)

- The RAG chatbot builds embeddings **at runtime** from its in-memory knowledge base using `all-MiniLM-L6-v2`. The large pre-built index referenced in older docs (`final_system/complete_index/`, "13,200 chunks") is **not present in this repository**; that figure is therefore **unverified** and should not be cited until the index is published.

## RAG architecture

```
Query ──► SentenceTransformer embedding (all-MiniLM-L6-v2)
      ──► cosine similarity vs. knowledge-base embeddings
      ──► Top-k retrieval (k=3, similarity threshold 0.05)
      ──► context assembly ──► response generation
```

Parameters read from `CreateModel/advanced_agricultural_rag_chatbot.py`:

| Setting | Value |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` (fallback: TF-IDF if unavailable) |
| Similarity | cosine similarity |
| Top-k | 3 |
| Threshold | 0.05 |

**Retrieval quality (recall@k, MRR) and response quality are not yet evaluated** — there is no retrieval-evaluation script committed. Do not cite retrieval-accuracy percentages until such a script and its output exist.

## Known failure modes & domain limits

- **Hallucination risk:** the generation step can produce fluent but unsupported agricultural advice, especially when retrieval returns low-similarity context.
- **Domain scope:** trained on six agricultural categories; out-of-domain queries are routed into the nearest category with possibly high confidence.
- **Language:** content mixes Turkish and English; behavior on other languages is untested.
- Predictions are decision-support, not agronomic ground truth.

## Installation

```bash
git clone https://github.com/Mertcan-Gelbal/agricultural-rag-edge-ai.git
cd agricultural-rag-edge-ai
pip install -r requirements.txt
cp .env.example .env        # add keys only if you enable external LLM calls
```

### Train a classifier (reproduce the measured results)

```bash
cd CreateModel
python3 train_distilbert.py        # DistilBERT
python3 train_bert_small.py        # BERT-small
```

### Run the RAG chatbot

```bash
pip install sentence-transformers scikit-learn pandas numpy
cd CreateModel
python3 advanced_agricultural_rag_chatbot.py
```

### Streamlit demo

```bash
streamlit run streamlit_app.py
```

## Jetson deployment

`setup_jetson62.sh` and `requirements_bert_jetpack62.txt` target JetPack 6.2 on Jetson Orin. The scripts install a CUDA-matched environment; **on-device inference latency and power draw are not yet measured**, so the per-model latency/power tables from earlier docs have been removed until benchmarked on hardware.

## Testing

Lightweight smoke tests live in `tests/` and cover dataset integrity and the retrieval helper's shape/threshold behavior (no GPU, no network, no API keys required):

```bash
pip install pytest
pytest -q
```

## Repository structure

```
Data/            Classification CSVs + combined pool + sentiment set + stats
CreateModel/     Training scripts, model dirs, RAG chatbot
Chatbot/         API server + CLI chatbots
jetson_training/ Jetson-oriented trainers
Model/           Exported small-BERT tokenizer/config
tests/           Smoke tests (data + retrieval)
streamlit_app.py Streamlit demo
```

> Legacy/duplicate trees (`project_structure_reorganized/`, multiple redundant chatbot scripts) remain from earlier iterations and are candidates for cleanup.

## Limitations

- RAG retrieval/response quality unmeasured; KB index not shipped.
- Only DistilBERT and BERT-small runs are reproducible from committed artifacts; larger-model claims removed.
- Jetson performance figures pending on-hardware measurement.
- Single-run metrics (no cross-seed averaging yet).

## Roadmap

- [ ] Add a fixed-seed `evaluate.py` that reports test-set accuracy/F1 and a classification report
- [ ] Add a retrieval-evaluation script (recall@k, MRR) and publish the KB index
- [ ] Measure Jetson latency/power per model and restore the table with device + precision documented
- [ ] Remove duplicate legacy trees

## License

MIT — see [LICENSE](./LICENSE).
