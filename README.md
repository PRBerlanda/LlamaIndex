# CSV Query Reader: ask questions about a CSV in plain Portuguese

Part of the [LlamaIndex](https://github.com/PRBerlanda/LlamaIndex) portfolio. Each project lives in its own branch.

This project lets you upload a CSV and ask questions about it in Portuguese. An LLM (Groq, through LlamaIndex) works in two steps:

1. It turns the question into a **pandas expression**, which runs on the DataFrame.
2. It reads the result and writes the answer, showing the code it used.

A Gradio app also keeps the question and answer history and exports it to PDF.

```
question ─► prompt 1 (column types + df.head()) ─► LLM ─► pandas expression
         ─► guarded eval on the DataFrame ─► result
         ─► prompt 2 (question + expression + result) ─► LLM ─► answer
```

| File | What it is |
|---|---|
| `csv_analyst.py` | The pipeline: prompts, code extraction, guarded execution, PDF export |
| `app.py` | Gradio app: upload, question, answer, PDF history, reset |
| `llm_config.py` | Groq client, configured from environment variables |
| `data/vendas.csv` | Sample data: 1,000 fictitious store sales (branch, customer type, product, total, payment method, rating) |
| `notebooks/` | The course notebook (Colab), kept with its original outputs |
| `tests/` | Offline tests (no API key needed) |

## How to run

```bash
git clone -b CSV-Query-Reader https://github.com/PRBerlanda/LlamaIndex.git csv-query-reader
cd csv-query-reader
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # put your free Groq key in it: https://console.groq.com/keys
set -a; source .env; set +a

python app.py                   # http://127.0.0.1:7860, then upload data/vendas.csv
```

You can also use the pipeline straight from Python:

```python
import pandas as pd
from csv_analyst import consultar
from llm_config import get_llm

r = consultar(pd.read_csv("data/vendas.csv"), "Qual é a avaliação média de cada filial?", get_llm())
print(r.expressao)   # df.groupby('filial')['avaliacao'].mean()
print(r.resposta)
```

## Tests

```bash
pytest tests/
```

The tests replace the LLM with a fake that returns the pandas code a good model would write. Each result is compared with an answer key computed directly in pandas. The tests also check that:

- the generated code is blocked when it tries to import modules, open files or reach private attributes;
- errors in the generated code are reported instead of crashing the app;
- the PDF accepts emojis and other characters outside Latin-1;
- the Groq settings are read from environment variables.

## Changes from the original course code

- **The pipeline no longer uses `QueryPipeline`.** It was removed in llama-index-core 0.13, so the same steps are now plain functions.
- **`PandasQueryEngine` / `PandasInstructionParser` were replaced** by a small evaluator that checks the generated code (AST) before running it.
  - `llama-index-experimental` is deprecated, and with current packages it no longer even imports: it fails on a `mistralai` import.
  - The checker is a guard rail, not a full sandbox. Only use the app with your own data, on your own machine.
- **Groq model updated:** `llama3-70b-8192` was shut down by Groq. The default is now `openai/gpt-oss-20b`, and you can change it with `GROQ_MODEL`.
- **API key** read from `GROQ_API_KEY`. `GROQ_API` also works, since that is the Colab secret name; the old app used `secret_key`.
- **App bugs fixed:**
  - "Gerar PDF" with an empty history returned two values to a single file component.
  - "Adicionar ao histórico" set the history to `None` when the question was empty.
- **PDF output:**
  - Moved from `fpdf` 1.7 to `fpdf2`, with `text=` and `new_x/new_y`.
  - Characters outside Latin-1 (emojis, long dashes) are replaced instead of crashing.
- **Gradio 6:** the theme moved to `launch()`.
- **Sample data** comes from the repository instead of a URL.

> The answers depend on the LLM and can vary from run to run. The live Groq API was not called during this review; the pipeline was checked with the fake LLM and against the pandas answer key.
