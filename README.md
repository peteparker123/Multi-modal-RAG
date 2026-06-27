# Multimodal RAG — PDF → ChromaDB + CLIP + Gemini

Two multimodal RAG pipelines for extracting and querying text, images, and tables from PDFs. Both share the same core stack but differ in how images are handled.

---

## Approaches

| | SmolVLM (v1) | CLIP Embeddings (v2) |
|---|---|---|
| **Image handling** | VLM generates a text description | CLIP encodes image directly to 512-d vector |
| **Image retrieval** | Text similarity (bge-small-en) | CLIP text-to-image similarity (FAISS) |
| **Accuracy** | Good for general queries | Better for visual similarity |
| **Speed** | Slower (VLM inference per image) | Fast (single forward pass) |
| **Notebook** | `Multi_modal_rag.ipynb` | `multimodal_rag_clip.ipynb` |

---

## Pipeline

```
PDF → Unstructured (text / images / tables)
       │
       ├── Text / Tables → bge-small-en → ChromaDB
       │
       └── Images → [SmolVLM description → ChromaDB]   (v1)
                    [CLIP ViT-B/32 → FAISS index    ]   (v2)
                              │
                    Query → dual retrieval → display visuals → Gemini answer
```

---

## Stack

| Component | Tool |
|-----------|------|
| PDF extraction | `unstructured[all-docs]` |
| Image description (v1) | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| Image embeddings (v2) | `openai/CLIP ViT-B/32` + FAISS |
| Text embeddings | `BAAI/bge-small-en` |
| Vector store | ChromaDB |
| LLM | Gemini 2.5 Flash |
| Framework | LangChain |

---

## Setup

Designed for **Google Colab** (GPU recommended for v1; CPU works fine for v2).

1. Set your PDF path:
   ```python
   PDF_PATH = "/content/your_file.pdf"
   ```
2. Set your Gemini API key:
   ```python
   GOOGLE_API_KEY = "your_key_here"
   ```
3. Run all cells top to bottom.

---

## Querying

```python
ask_multimodal("Show me the weapon samples from the dataset")
ask_multimodal("What does the training loss curve show?")
ask_multimodal("What is the mAP50 for the knife class?")
```

Matching images and tables are **displayed automatically** before the text answer.

**v2 only — image-to-image search:**
```python
results = clip_image_to_image_search("/path/to/query.jpg", top_k=3)
```

---

## Metadata Stored per Chunk

| Field | Images | Tables | Text |
|-------|--------|--------|------|
| `source` | `"image"` | `"table"` | `"text"` |
| `image_path` | ✓ | ✓ | — |
| `page_number` | ✓ | ✓ | ✓ |
| `element_index` | ✓ | ✓ | — |
| `raw_text` | — | ✓ | — |

---

## Notes

- Tables use unstructured's extracted text directly — no VLM needed in either version
- ChromaDB is wiped and rebuilt on each run to avoid duplicate docs
- v1 uses `repetition_penalty=1.5` and `no_repeat_ngram_size=4` to prevent SmolVLM looping
- v2 CLIP index is saved to disk (`clip_index.faiss` + `clip_meta.pkl`) so images aren't re-encoded on every run
- v2 runs dual retrieval — ChromaDB for text/tables, FAISS for images — and merges context before the LLM call
