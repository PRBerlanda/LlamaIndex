"""Gradio chat for the Serenatto coffee shop.

    export GROQ_API_KEY=...      # see .env.example
    python app.py                # http://127.0.0.1:7860
"""
import gradio as gr

from llm_config import get_llm
from serenatto_chatbot import construir_indice, criar_chat_engine, embedding_padrao

_estado = {}


def chat_engine():
    if "engine" not in _estado:
        _estado["engine"] = criar_chat_engine(construir_indice(embedding_padrao()), get_llm())
    return _estado["engine"]


def converse_com_bot(message, chat_history):
    chat_history = list(chat_history or [])
    if not message:
        return "", chat_history
    try:
        resposta = chat_engine().chat(message).response
    except Exception as e:
        resposta = f"Desculpe, ocorreu um erro: {e}"
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": resposta})
    return "", chat_history


def resetar_chat():
    if "engine" in _estado:
        _estado["engine"].reset()
    return []


with gr.Blocks(title="Chatbot da Serenatto") as app:
    gr.Markdown("# Chatbot da Serenatto ☕")
    chatbot = gr.Chatbot(label="Assistente Serenatto")
    msg = gr.Textbox(label="Digite a sua mensagem", placeholder="Quais grãos estão disponíveis?")
    limpar = gr.Button("Limpar")
    msg.submit(converse_com_bot, [msg, chatbot], [msg, chatbot])
    limpar.click(resetar_chat, None, chatbot, queue=False)

if __name__ == "__main__":
    app.launch()
