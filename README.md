# SQL Query Reader: Text-to-SQL assistant for an e-commerce database

Part of the [LlamaIndex](https://github.com/PRBerlanda/LlamaIndex) portfolio. Each project lives in its own branch.

This is a chat assistant that answers questions about **Zoop**, a fictitious online store, by writing and running SQL on a SQLite database. It has three tables:

- `Clientes` (customers): 100 rows;
- `Fornecedores` (suppliers): 6 rows;
- `Funcionarios` (employees): 7 rows.

```
question ─► table retriever (embeddings of LLM-written table descriptions)
         ─► table schema + description ─► prompt 1 ─► LLM ─► SQL
         ─► read-only check ─► SQLite ─► rows ─► prompt 2 ─► LLM ─► answer (Gradio chat)
```

| File | What it is |
|---|---|
| `sql_assistant.py` | Table descriptions, table retriever, Text-to-SQL prompts, SQL extraction and validation |
| `app.py` | Gradio chat |
| `llm_config.py` | Groq client, configured from environment variables |
| `data/ecommerce.db` | Sample SQLite database. All data is fictitious, for example `nome@email.com` |
| `data/descricoes_tabelas.json` | Table descriptions written by the LLM in the course run, cached so the app does not regenerate them |
| `notebooks/` | The course notebook (Colab), kept with its original outputs |
| `tests/` | Offline tests |

## How to run

```bash
git clone -b SQL-Query-Reader https://github.com/PRBerlanda/LlamaIndex.git sql-query-reader
cd sql-query-reader
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt     # CPU-only PyTorch is enough

cp .env.example .env                # your free Groq key: https://console.groq.com/keys
set -a; source .env; set +a

python app.py                       # http://127.0.0.1:7860
```

Example questions:

- "Quais estados mais frequentes na tabela Clientes?"
- "Quais as fornecedoras localizadas na cidade de São Paulo?"
- "Quantos funcionários com o cargo de vendedores temos cadastrados?"

The first run downloads the embedding model `BAAI/bge-m3` (about 2.3 GB). Set `EMBED_MODEL=intfloat/multilingual-e5-small` for a faster test. Delete `data/descricoes_tabelas.json` if you want the LLM to describe the tables again.

## Tests

```bash
pytest tests/
```

The tests need no download, API key or internet. A fake LLM returns the SQL a good model would write for the three course questions, and the result is compared with an answer key computed directly with `sqlite3`: 3 sellers, and 2 suppliers in São Paulo. They also check that:

- the schema and description of the chosen table reach the prompt;
- SQL wrapped in ```` ```sql ```` fences is extracted correctly;
- `DROP`, `DELETE`, `UPDATE` and multiple statements are blocked, and the database is unchanged;
- the Gradio handler returns messages in the `role` / `content` format.

## Changes from the original course code

- **The pipeline no longer uses `QueryPipeline`.** It was removed in llama-index-core 0.13, so the same steps (`FnComponent`, `SQLRetriever`, two prompts) are now plain functions.
- **SQL extraction fixed.** `strip('```')` left the language tag when the model answered with ```` ```sql ... ``` ````, so the query started with `sql\nSELECT` and failed.
- **Read-only guard added.** Only a single `SELECT`/`WITH` statement is executed, because the LLM must never change the database. SQL errors are passed to the second prompt instead of crashing the chat.
- **Groq model updated:** `llama-3.1-70b-versatile` was shut down. The default is now `openai/gpt-oss-20b`, configurable with `GROQ_MODEL`.
- **Gradio 6:** the chat history uses `{"role", "content"}` messages; the course's `[user, bot]` pairs no longer work.
- **Table descriptions are cached** in `data/descricoes_tabelas.json`, taken from the course run. The prompt now also says that text values are lowercase and without accents (`'sao paulo'`, `'vendas'`), which is what made the first version answer "no suppliers in São Paulo".
- **Notebook cleanup.** The empty exercise template `LLM_assistant_SQL.ipynb` was not uploaded; only the completed notebook is kept.

> The answers depend on the LLM and can vary from run to run. The live Groq API and the embedding model were not called during this review; the pipeline was tested offline against the SQLite answer key.
