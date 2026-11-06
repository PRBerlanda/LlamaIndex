"""Gradio app: upload a CSV, ask questions in Portuguese, export the Q&A history to PDF.

    export GROQ_API_KEY=...      # see .env.example
    python app.py                # http://127.0.0.1:7860
"""
import tempfile

import gradio as gr
import pandas as pd

from csv_analyst import consultar, gerar_pdf

_llm = None


def llm():
    """Create the Groq client on first use, so the UI starts even without a key."""
    global _llm
    if _llm is None:
        from llm_config import get_llm

        _llm = get_llm()
    return _llm


def carregar_dados(caminho_arquivo, df_estado):
    if not caminho_arquivo:
        return "Por favor, faça o upload de um arquivo CSV para analisar.", pd.DataFrame(), df_estado
    try:
        df = pd.read_csv(caminho_arquivo)
        return "Arquivo carregado com sucesso!", df.head(), df
    except Exception as e:
        return f"Erro ao carregar arquivo: {e}", pd.DataFrame(), df_estado


def processar_pergunta(pergunta, df_estado):
    if df_estado is None or not pergunta:
        return ""
    try:
        return consultar(df_estado, pergunta, llm()).resposta
    except Exception as e:
        return f"Erro: {e}"


def add_historico(pergunta, resposta, historico_estado):
    historico_estado = list(historico_estado or [])
    if pergunta and resposta:
        historico_estado.append((pergunta, resposta))
        gr.Info("Adicionado ao PDF!", duration=2)
    return historico_estado


def gerar_pdf_app(historico_estado):
    caminho = gerar_pdf(historico_estado, tempfile.gettempdir())
    if caminho is None:
        gr.Warning("Nenhum dado para adicionar ao PDF.")
        return None
    return str(caminho)


def limpar_pergunta_resposta():
    return "", ""


def resetar_aplicacao():
    return (None, "A aplicação foi resetada. Por favor, faça upload de um novo arquivo CSV.",
            pd.DataFrame(), "", None, [], "")


with gr.Blocks(title="Analisando os dados") as app:
    gr.Markdown("# Analisando os dados 🔎🎲")
    gr.Markdown(
        """
        Carregue um arquivo CSV e faça perguntas sobre os dados. A cada pergunta, você pode
        adicionar a interação ao PDF final em "Adicionar ao histórico do PDF". Para uma nova pergunta,
        clique em "Limpar pergunta e resultado". Depois clique em "Gerar PDF" para baixar o registro.
        Para analisar outro dataset, use o botão no final da página.
        """
    )
    input_arquivo = gr.File(file_count="single", type="filepath", label="Upload CSV")
    upload_status = gr.Textbox(label="Status do Upload:")
    tabela_dados = gr.DataFrame()
    gr.Markdown(
        """
        Exemplos de perguntas:
        1. Qual é o número de registros no arquivo?
        2. Quais são os tipos de dados das colunas?
        3. Qual é a forma de pagamento mais utilizada pelos clientes?
        """
    )
    input_pergunta = gr.Textbox(label="Digite sua pergunta sobre os dados")
    botao_submeter = gr.Button("Enviar")
    output_resposta = gr.Textbox(label="Resposta")
    with gr.Row():
        botao_limpeza = gr.Button("Limpar pergunta e resultado")
        botao_add_pdf = gr.Button("Adicionar ao histórico do PDF")
        botao_gerar_pdf = gr.Button("Gerar PDF")
    arquivo_pdf = gr.File(label="Download do PDF")
    botao_resetar = gr.Button("Quero analisar outro dataset!")

    df_estado = gr.State(value=None)
    historico_estado = gr.State(value=[])

    input_arquivo.change(carregar_dados, [input_arquivo, df_estado], [upload_status, tabela_dados, df_estado])
    botao_submeter.click(processar_pergunta, [input_pergunta, df_estado], output_resposta)
    botao_limpeza.click(limpar_pergunta_resposta, [], [input_pergunta, output_resposta])
    botao_add_pdf.click(add_historico, [input_pergunta, output_resposta, historico_estado], historico_estado)
    botao_gerar_pdf.click(gerar_pdf_app, [historico_estado], arquivo_pdf)
    botao_resetar.click(
        resetar_aplicacao, [],
        [input_arquivo, upload_status, tabela_dados, output_resposta, arquivo_pdf, historico_estado, input_pergunta],
    )

if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft())
