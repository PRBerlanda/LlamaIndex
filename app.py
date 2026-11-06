"""Gradio chat with the Zoop SQL assistant.

    export GROQ_API_KEY=...      # see .env.example
    python app.py                # http://127.0.0.1:7860
"""
import gradio as gr

from llm_config import get_llm
from sql_assistant import embedding_padrao, montar_assistente, responder

_estado = {}


def assistente():
    if not _estado:
        llm = get_llm()
        sql_database, retriever = montar_assistente(llm, embedding_padrao())
        _estado.update(llm=llm, sql_database=sql_database, retriever=retriever)
    return _estado


def entrada_saida(msg_user: str) -> str:
    a = assistente()
    return responder(msg_user, a["llm"], a["sql_database"], a["retriever"]).resposta


def atualizar_historico(msg_user, historico):
    historico = list(historico or [])
    if not msg_user:
        return "", historico
    try:
        msg_assistente = entrada_saida(msg_user)
    except Exception as e:
        msg_assistente = f"Erro: {e}"
    historico.append({"role": "user", "content": msg_user})
    historico.append({"role": "assistant", "content": msg_assistente})
    return "", historico


with gr.Blocks(title="Assistente SQL Zoop") as app:
    gr.Markdown("## Chat com Assistente SQL")
    gr.Markdown(
        """
        Assistente SQL interativo que responde perguntas sobre os dados da loja Zoop
        (tabelas Clientes, Fornecedores e Funcionarios), com base no resultado de consultas SQL.
        """
    )
    chatbot = gr.Chatbot(label="Chat com Assistente")
    msg = gr.Textbox(label="Digite a sua pergunta e tecle Enter para enviar",
                     placeholder="Quantos funcionários com o cargo de vendedores temos cadastrados?")
    limpeza = gr.Button("Limpar a conversa")
    msg.submit(atualizar_historico, inputs=[msg, chatbot], outputs=[msg, chatbot])
    limpeza.click(lambda: [], inputs=None, outputs=chatbot, queue=False)

if __name__ == "__main__":
    app.launch()
