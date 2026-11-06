"""Gradio app for the Buscante book recommender.

    export GROQ_API_KEY=...      # see .env.example
    python app.py                # http://127.0.0.1:7860

The first run downloads the embedding model and embeds the catalogue + articles
into ./chroma_db (several minutes on CPU); later runs reuse that folder.
"""
import gradio as gr
import pandas as pd

from book_recommender import CATALOGO, RAGStringQueryEngine, construir_indice, embedding_padrao, pipeline_completo
from llm_config import get_llm

_estado = {}


def recursos():
    if not _estado:
        llm = get_llm(max_tokens=4096)
        index = construir_indice(embedding_padrao())
        _estado.update(
            llm=llm,
            df=pd.read_csv(CATALOGO),
            engine=RAGStringQueryEngine(retriever=index.as_retriever(similarity_top_k=5), llm=llm),
        )
    return _estado


def recomendar(consulta: str) -> str:
    if not consulta:
        return ""
    r = recursos()
    try:
        return pipeline_completo(consulta, r["engine"], r["llm"], r["df"])
    except Exception as e:
        return f"Não consegui gerar uma recomendação válida: {e}"


app = gr.Interface(
    fn=recomendar,
    inputs=gr.Textbox(label="O que você gostaria de ler?", placeholder="Quais livros de romance vocês têm?"),
    outputs=gr.Textbox(label="Recomendação"),
    title="Especialista Buscante",
    flagging_mode="never",
)

if __name__ == "__main__":
    app.launch()
