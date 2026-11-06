# LlamaIndex – LLM applications with Groq

Projects built while studying **LlamaIndex** (Alura course track). They are LLM applications that answer questions about different kinds of data: a CSV, a SQL database, a book catalogue and a PDF. All of them use **Groq** models and **Gradio** interfaces.

**Each project lives in its own branch.** This `main` branch only holds this index.

| Branch | Project | Highlights |
|---|---|---|
| [`CSV-Query-Reader`](https://github.com/PRBerlanda/LlamaIndex/tree/CSV-Query-Reader) | Ask questions about a CSV | LLM → pandas code → guarded execution → answer, PDF report of the Q&A |
| [`SQL-Query-Reader`](https://github.com/PRBerlanda/LlamaIndex/tree/SQL-Query-Reader) | Text-to-SQL assistant | Table retrieval by embeddings, SQL generation, read-only guard, chat |
| [`Automated-Response-JSON-Formatting`](https://github.com/PRBerlanda/LlamaIndex/tree/Automated-Response-JSON-Formatting) | Book recommender | RAG with Chroma, JSON-only answers, validation against the catalogue |
| [`RAG-Chatbot`](https://github.com/PRBerlanda/LlamaIndex/tree/RAG-Chatbot) | Coffee-shop chatbot | PDF → chunks → Chroma, chat engine with memory, Langflow version and web page |

## Reproducing a project

Every branch is self-contained and includes:

- `README.md`: what the project does, how to run it, and the changes made to the original course code;
- `requirements.txt`: pinned dependencies, tested with Python 3.13 (virtual environments are not committed);
- `.env.example`: settings; the Groq key is read from the environment and never stored in the code;
- `tests/`: offline checks with `pytest`, using a fake LLM and embedding, so no API key or model download is needed;
- the sample data used by the project, plus the original course notebook with its outputs.

To install everything at once, use the `requirements.txt` in this `main` branch. Each branch also has its own, smaller one.

```bash
git clone -b RAG-Chatbot https://github.com/PRBerlanda/LlamaIndex.git
cd LlamaIndex
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest tests/
```

## Stack

LlamaIndex 0.14 · Groq (`openai/gpt-oss-20b`) · Hugging Face embeddings (multilingual E5, BGE-M3) · ChromaDB · pandas · SQLAlchemy/SQLite · Gradio 6 · Langflow.
