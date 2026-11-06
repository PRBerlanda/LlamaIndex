# Automated responses with JSON formatting: a book recommender with RAG

Part of the [LlamaIndex](https://github.com/PRBerlanda/LlamaIndex) portfolio. Each project lives in its own branch.

This is a recommendation assistant for **Buscante**, a fictitious bookstore. The LLM may only recommend books the store actually sells:

1. The best-seller catalogue (175 books) and 41 short articles, one per category, are embedded with a multilingual **E5** model and stored in **Chroma**.
2. Each question retrieves the closest chunks. The LLM must answer **only with JSON**: title, category and reason for each book.
3. The JSON is parsed, and **every title is checked against the catalogue**. If a title is not there, the answer is rejected.
4. A second LLM call turns the validated JSON into a friendly reply, shown in a Gradio app.

| File | What it is |
|---|---|
| `book_recommender.py` | Index (Chroma), RAG query engine, JSON parsing, validation, final formatting |
| `app.py` | Gradio app |
| `llm_config.py` | Groq client, configured from environment variables |
| `data/livros_mais_vendidos.csv` | Catalogue: `nome`, `categoria`, `sinopse`, `preco` |
| `data/artigos/` | One article per category, mentioning catalogue titles |
| `notebooks/` | The course notebook (Colab), kept with its original outputs |
| `tests/` | Offline tests |

## How to run

```bash
git clone -b Automated-Response-JSON-Formatting https://github.com/PRBerlanda/LlamaIndex.git book-recommender
cd book-recommender
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt     # CPU-only PyTorch is enough

cp .env.example .env                # your free Groq key: https://console.groq.com/keys
set -a; source .env; set +a

python app.py                       # http://127.0.0.1:7860
```

The first run takes a while:

- the embedding model `intfloat/multilingual-e5-large` (about 2.2 GB) is downloaded;
- every document is embedded into `./chroma_db`, which takes several minutes on CPU.

Later runs reuse `./chroma_db`. Set `EMBED_MODEL=intfloat/multilingual-e5-small` for a much faster test, then delete `chroma_db/` whenever you change the model.

## Tests

```bash
pytest tests/
```

The tests need no download, API key or internet. A keyword-hashing embedding replaces E5 and a fake LLM replaces Groq. They check that:

- the index holds both the catalogue and the articles, and a romance query retrieves the romance article;
- the full pipeline only recommends catalogue books;
- JSON is extracted from answers wrapped in ```` ```json ```` fences or extra text;
- validation rejects an invented title even when it is the *first* item;
- titles with parentheses validate correctly;
- a persisted index is reopened without embedding again.

## Changes from the original course code

- **Validation fixed in two places:**
  - `validando()` had the `if` outside the `for` loop, so only the **last** title was checked.
  - It used `str.contains()`, which reads titles such as `... (EDIÇÃO DE COLECIONADOR)` as regular expressions. Validation now uses an exact, case-insensitive match.
- **JSON parsing is more tolerant.** `json.loads()` failed whenever the model wrapped the answer in ```` ```json ```` fences or added a sentence. The parser now extracts the object first.
- **The index is reopened correctly.** `load_index_from_storage()` on a Chroma storage context fails in a new session, because there is no persisted docstore. The index is now reopened with `VectorStoreIndex.from_vector_store()`, so documents are embedded only once.
- **Removed the `ChromaEmbeddingWrapper`.**
  - It called `HuggingFaceEmbedding.embed()`, which does not exist.
  - It loaded the 2.2 GB model a second time.
  - LlamaIndex already sends the vectors to Chroma.
- **E5 prefixes added.** The E5 model is now used with its expected `query: ` / `passage: ` prefixes, which improves retrieval.
- **Folder names fixed.** The notebook read from `csv/` and `artigos/`, which did not match the files (`Artigos/`, CSV in the root). Paths now come from `data/`.
- **Groq model updated:** `llama3-70b-8192` was shut down. The default is now `openai/gpt-oss-20b`, configurable with `GROQ_MODEL`.
- **The course notebook is kept as a record**, with its original outputs; only the retired model ID was changed.

> The answers depend on the LLM and can vary from run to run. The live Groq API and the E5 model were not called during this review; the pipeline was tested offline.
