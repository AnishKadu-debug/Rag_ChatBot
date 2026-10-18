# Evaluation & Tuning Guide (Phase 4)

This folder contains the evaluation dataset, evaluation runner, and benchmarking tools for the WhatsApp Land Records RAG pipeline.

---

## 1. Evaluation Dataset (`eval/questions.json`)

The evaluation dataset contains 23 realistic test questions strictly grounded in the sample land-record PDF (`data/sample_pdfs/sample_land_record.pdf`):

| Category | Count | Description | Expected Page(s) |
| :--- | :--- | :--- | :--- |
| **explicit** | 12 | Direct factual questions explicitly stated in the document (owner name, Gat number, total area, khata, cultivable area, poti-kharaba, crop, mutation number, date, waras type, talathi certification, demise of previous holder). | Page 1 or Page 2 |
| **multi_page** | 3 | Questions requiring retrieval and synthesis of facts across both physical pages (e.g. Village/Taluka/District location across 7/12 and Ferfar; complete ownership succession timeline). | Pages [1, 2] |
| **paraphrased** | 4 | Questions formulated with alternative phrasing, synonyms, or cross-lingual Marathi queries (e.g. "title holder of agricultural parcel", "produce farmed during monsoon", Marathi query `या जमिनीचे एकूण क्षेत्रफळ आणि वहितीखालील जमीन किती आहे?`). | Page 1 or Page 2 |
| **absent** | 4 | Questions whose answers are not in the document (land tax/cess, bank loan/mortgage, ready reckoner rates, and an off-topic recipe question). | `[]` (Fallback expected) |

Every entry specifies:
- `id`: Unique identifier (e.g., `q01`).
- `question`: Natural user question text.
- `category`: Classification (`explicit`, `multi_page`, `paraphrased`, `absent`).
- `is_answerable`: Boolean indicating if the document contains the fact.
- `expected_pages`: List of 1-based physical page numbers (`[1]`, `[2]`, `[1, 2]`, or `[]`).
- `expected_answer`: Ground truth answer summary.
- `expected_keywords`: Key terms that must be present in a grounded response.
- `notes`: Document grounding verification notes.

---

## 2. Evaluation Runner (`eval/run_eval.py`)

### Usage

Run from the project root with the virtual environment activated:

```bash
# 1. Run basic retrieval evaluation
python eval/run_eval.py

# 2. Run retrieval evaluation with verbose per-question output
python eval/run_eval.py -v

# 3. Run retrieval evaluation AND systematic parameter tuning grid
python eval/run_eval.py --tune

# 4. Save results to a JSON file
python eval/run_eval.py --output eval/eval_results.json

# 5. Run with real Gemini LLM answer generation (when API key is provided)
python eval/run_eval.py --api-key YOUR_GEMINI_KEY
# or configure GEMINI_API_KEY in your .env file
```

### Command-Line Arguments

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--questions` | Path | `eval/questions.json` | Path to evaluation questions file |
| `--pdf` | Path | `data/sample_pdfs/sample_land_record.pdf` | Path to target sample PDF |
| `--top-k` | int | `4` | Number of chunks retrieved per query |
| `--threshold` | float | `0.45` | Cosine distance threshold |
| `--chunk-size` | int | `500` | Target character size per chunk |
| `--chunk-overlap`| int | `100` | Overlap character size |
| `--retrieval-only`| flag | `False` | Run only retrieval evaluation, skipping LLM answer generation |
| `--tune` | flag | `False` | Run systematic tuning grid sweep across parameter combinations |
| `--api-key` | str | `None` | Gemini API key override (never logged or stored in files) |
| `--output` | Path | `None` | Optional path to export JSON evaluation report |
| `-v`, `--verbose`| flag | `False` | Print question-by-question breakdown |

---

## 3. Metrics & Interpretation

### A. Core Retrieval Metrics
1. **Top-1 Hit Rate (P@1)**:
   - Percentage of answerable questions where the rank 1 retrieved chunk matches the expected source page.
   - Target: > 90%. Benchmark achieved: **100.0%**.
2. **Top-k Hit Rate (R@k)**:
   - Percentage of answerable questions where at least one chunk in top-k matches the expected source page.
   - Target: > 95%. Benchmark achieved: **100.0%**.
3. **Mean Reciprocal Rank (MRR)**:
   - Average of the reciprocal rank `1 / rank` for the first relevant retrieved chunk. An MRR of 1.0 means the relevant page was always at rank 1.
   - Benchmark achieved: **1.0000**.
4. **Multi-page Coverage**:
   - For questions requiring information from multiple pages (e.g. `[1, 2]`), the percentage of queries where *all* expected pages appear in top-k.
   - Benchmark achieved: **100.0%** at `top_k=4`.

### B. Cosine Distance & Gatekeeper Threshold
In cosine distance:
- `0.0`: Exact match.
- `0.10 - 0.25`: Highly relevant semantic match.
- `> 0.45`: Weak relevance or off-topic.

**Empirical Distance Distribution on Sample Land Record:**
- Answerable questions: Min = `0.1391`, Mean = `0.1927`, Max = `0.2398`.
- Absent / Off-topic questions: Min = `0.1888`, Mean = `0.2127`, Max = `0.2344`.

**Key Finding on Small Datasets:**
With very small document sets (2 pages / 2 chunks), 768-dimensional normalized embeddings naturally produce cosine distances around 0.19 to 0.24 even for off-topic questions.
- A tight threshold like `0.20` causes **false negatives** (blocking 52.6% of valid answerable queries like cultivable area or owner name).
- Therefore, the distance threshold is kept at `0.45` to guarantee **100% recall** on answerable queries, while domain-adjacent absent facts are safely handled by the LLM prompt's `NOT_FOUND` token and general fallback mechanism.

### C. LLM Answer Evaluation
When `GEMINI_API_KEY` is provided:
- **Accuracy**: Grounded answers containing all expected keywords and valid page citations.
- **Citation Accuracy**: Verifies whether cited pages (`(Page 1)`) strictly match expected pages.
- **Incomplete**: Flagged when an answer is in document mode but omits key facts or numbers.
- **Unsupported**: Flagged when an answer hallucinates facts or cites the wrong page.
- **Absent Fallback Accuracy**: Verifies that absent/out-of-scope questions fall back to `mode="general"`.

When `GEMINI_API_KEY` is not set:
- The script clearly outputs `[LLM EVALUATION SKIPPED]` with explanation.
- Per project rules, `MockLLMClient` is **never** used to fake real LLM quality.

---

## 4. Tuning Grid Summary

| Chunk/Overlap | Total Chunks | Top-K | Distance Threshold | Hit@1 | Hit@K | MRR | Multi-page Coverage | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **500 / 100** (Default) | 2 | 2 | 0.45 | 100.0% | 100.0% | 1.0000 | 100.0% | 90.5% |
| **500 / 100** (Default) | 2 | 4 | 0.45 | 100.0% | 100.0% | 1.0000 | 100.0% | 90.5% |
| **500 / 100** | 2 | 4 | 0.35 | 100.0% | 100.0% | 1.0000 | 100.0% | 90.5% |
| **500 / 100** | 2 | 4 | 0.55 | 100.0% | 100.0% | 1.0000 | 100.0% | 90.5% |
| **250 / 50** | 4 | 2 | 0.45 | 100.0% | 100.0% | 1.0000 | **33.3%** ⚠️ | 90.5% |
| **250 / 50** | 4 | 4 | 0.45 | 100.0% | 100.0% | 1.0000 | 100.0% | 90.5% |
| **250 / 50** | 4 | 4 | 0.35 | 100.0% | 100.0% | 1.0000 | 100.0% | 90.5% |
| **250 / 50** | 4 | 4 | 0.55 | 100.0% | 100.0% | 1.0000 | 100.0% | 90.5% |

**Tuning Conclusion**:
- Retaining `CHUNK_SIZE=500`, `CHUNK_OVERLAP=100`, `TOP_K=4`, and `SIMILARITY_DISTANCE_THRESHOLD=0.45` is empirically justified:
  1. Smaller chunks (`250/50`) with `top_k=2` degrade multi-page coverage from 100% down to **33.3%** because multiple chunks from one page push chunks from other required pages out of the retrieval window.
  2. Larger chunks (`500/100`) preserve full page context without fragmentation.
  3. `TOP_K=4` guarantees full multi-page coverage across both configurations.
