# RAG Chatbot: Serenatto specialty coffee assistant

Part of the [LlamaIndex](https://github.com/PRBerlanda/LlamaIndex) portfolio. Each project lives in its own branch.

This is a customer-service chatbot for **Serenatto**, a fictitious online store of specialty coffee. It answers from the store's FAQ and product PDF using **Retrieval-Augmented Generation (RAG)**:

```
PDF ─► text cleaning ─► chunks (SentenceSplitter, 1200 tokens) ─► embeddings (multilingual E5)
    ─► Chroma ─► chat engine "context" mode + summarised memory (Groq LLM) ─► Gradio chat
```

The branch also has the **Langflow** version of the same bot and the store's web page, which embeds the Langflow chat widget.

| File | What it is |
|---|---|
| `serenatto_chatbot.py` | PDF loading and cleaning, chunking, Chroma index, chat engine |
| `app.py` | Gradio chat |
| `llm_config.py` | Groq client, configured from environment variables |
| `data/serenatto_cafes_especiais.pdf` | Source document: FAQ, the six coffees, shipping and payment |
| `langflow/simple_agent_flow.json` | Langflow flow export (import it in Langflow) |
| `langflow/LangflowSchema.png` | Diagram of the flow |
| `web/index.html` + `web/images/` | Store landing page with the `langflow-chat` widget |
| `notebooks/` | The course notebook (Colab), kept with its original outputs |
| `tests/` | Offline tests |

## How to run

```bash
git clone -b RAG-Chatbot https://github.com/PRBerlanda/LlamaIndex.git rag-chatbot
cd rag-chatbot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt     # CPU-only PyTorch is enough

cp .env.example .env                # your free Groq key: https://console.groq.com/keys
set -a; source .env; set +a

python app.py                       # http://127.0.0.1:7860
```

Try "Quais grãos estão disponíveis?", then "E o Geisha, como é?". The chat keeps the context of the conversation.

The first run downloads `intfloat/multilingual-e5-large` (about 2.2 GB) and stores the vectors in `./chroma_db`. Set `EMBED_MODEL=intfloat/multilingual-e5-small` for a faster test, and delete `chroma_db/` when you change the model.

### Langflow version and web page

1. Run Langflow (`pip install langflow && langflow run`).
2. Import `langflow/simple_agent_flow.json`.
3. Fill in the components' credentials. The flow uses OpenAI embeddings and an Oracle vector store, so you can swap them for the providers you have.
4. Copy the flow id, then set `flow_id` and `host_url` of the `<langflow-chat>` tag in `web/index.html`.
5. Open `web/index.html` in the browser.

## Tests

```bash
pytest tests/
```

The tests need no download, API key or internet. A keyword-hashing embedding replaces E5 and a fake LLM replaces Groq. They check that:

- the PDF becomes 10 clean chunks that mention every coffee;
- a question about the Geisha retrieves the Geisha chunk;
- the chat engine sends the system prompt and context to the LLM and keeps or resets the history;
- the Gradio handlers return messages in the `role` / `content` format.

## Changes from the original course code

- **PDF text cleaned before chunking.** pypdf extracts this PDF with a line break between every word (`Loja\n \nonline`). The whitespace is now collapsed, so each chunk holds real sentences and the embeddings are not diluted by blank tokens.
- **Vector inspection fixed.** The notebook embedded the LLM's *answer* when looking at the closest vectors (`get_text_embedding(response)`). `trechos_relevantes()` now uses the *question*.
- **The index is reopened correctly.** `load_index_from_storage()` on a Chroma storage context fails in a new session, so the index is now reopened with `VectorStoreIndex.from_vector_store()`.
- **Removed the `ChromaEmbeddingWrapper`.** It called a method that does not exist (`.embed`) and loaded the model twice. The E5 model now uses its `query: ` / `passage: ` prefixes.
- **The web page is self-contained.**
  - It loaded the images from the instructor's GitHub repository; it now uses the local `web/images/`, resized from about 3 MB to about 0.5 MB each.
  - It pointed to the instructor's Langflow Space; that is now a placeholder for your own flow.
- **Updated libraries:** llama-index-core 0.14 and Gradio 6. The chat uses the `messages` format.
- **Groq configuration:** the key and model come from environment variables (`GROQ_API_KEY`, `GROQ_MODEL`) instead of Colab secrets.

> The answers depend on the LLM and can vary from run to run. The live Groq API, the E5 model and the Langflow flow were not run during this review; the Python pipeline was tested offline.
