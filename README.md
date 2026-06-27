# Multimodal RAG — Unstructured + SmolVLM + ChromaDB

A multimodal RAG pipeline that extracts text, images, and tables from PDFs, describes visuals using SmolVLM, stores everything in ChromaDB with metadata, and retrieves + displays the right image or table when you ask a question.

---

## How It Works

```
PDF → Unstructured (extract text / images / tables)
     → SmolVLM-256M (describe images)
     → ChromaDB (embed + store with metadata)
     → Query → retrieve → display image/table + LLM answer
```

---

## Stack

| Component | Tool |
|-----------|------|
| PDF extraction | `unstructured[all-docs]` |
| Image description | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| Embeddings | `BAAI/bge-small-en` |
| Vector store | ChromaDB |
| LLM | Gemini 2.5 Flash |
| Framework | LangChain |

---

## Setup

Designed to run on **Google Colab** (GPU recommended).

1. Clone the repo and open the notebook in Colab
2. Set your PDF path:
   ```python
   PDF_PATH = "/content/your_file.pdf"
   ```
3. Set your Gemini API key:
   ```python
   GOOGLE_API_KEY = "your gemini api key"
   ```
4. Run all cells top to bottom

---

## Query

```python
ask_multimodal("Show me the weapon samples from the dataset")
ask_multimodal("What does the training loss curve show?")
ask_multimodal("What is the mAP50 for the knife class?")
```

When the answer involves an image or table, it is **displayed automatically** before the text answer.

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

- Tables use unstructured's extracted text directly — no VLM needed
- ChromaDB is wiped and rebuilt cleanly on each run to avoid duplicate docs
- SmolVLM generation uses `repetition_penalty=1.5` and `no_repeat_ngram_size=4` to prevent looping
- Retrieval runs two passes — one mixed, one image/table-only — so visuals are never crowded out by text chunks
