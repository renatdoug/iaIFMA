import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import csv
import time

def formatar_diagnostico(diagnostico):
    return diagnostico.replace('_', ' ')

def main():
    st.markdown("<h3 style='text-align: center; color: #FF5733;'>Sistema de Previsão de Diagnóstico de Enfermagem</h3>", unsafe_allow_html=True)

    # Carrega o DataFrame
    df = pd.read_csv('Dados.csv')
    atributos = [coluna for coluna in df.columns if coluna != 'diagnostico_de_Enfermagem']

    # Carrega o modelo
    with open('modelo_4.pkl', 'rb') as file:
        best_tree = pickle.load(file)

    # Carrega os cuidados relacionados aos diagnósticos
    cuidados_df = pd.read_csv('cuidados_diags.csv')

    # Barra lateral para entrada das iniciais do usuário
    st.sidebar.header('Entrada com suas iniciais')
    iniciais_usuario = st.sidebar.text_input('Iniciais do Usuário')

    # Seleção dos sintomas
    sintomas = st.multiselect("Selecione os sintomas:", atributos)

    diagnosticos_selecionados = {}

    if sintomas:
        diagnosticos_associados = []
        limiar_probabilidade = 0.1

        label_encoder = LabelEncoder()
        label_encoder.fit(df['diagnostico_de_Enfermagem'])

        # Registro do tempo de início da seleção de sintomas
        inicio_secao = time.time()

        for sintoma in sintomas:
            dados_paciente_dict = {coluna: 0 for coluna in atributos}

            if sintoma in dados_paciente_dict:
                dados_paciente_dict[sintoma] = 1

            dados_paciente_np = [dados_paciente_dict[coluna] for coluna in atributos]

            probas = best_tree.predict_proba([dados_paciente_np])[0]

            indices_diagnosticos = [indice for indice, probabilidade in enumerate(probas) if probabilidade > limiar_probabilidade]
            for indice in indices_diagnosticos:
                diagnostico = label_encoder.inverse_transform([indice])[0]
                diagnosticos_associados.append((diagnostico, sintoma))

        # Exibição dos diagnósticos sugeridos
        st.write("Diagnósticos sugeridos com base nos sintomas selecionados:")

        for diagnostico, sintoma in diagnosticos_associados:
            sintoma_selecionado = sintoma.replace("_", " ")
            sintoma_selecionado = sintoma_selecionado.replace("-", " ")  # Substituir traços por espaços
            sintoma_selecionado = sintoma_selecionado.lower()  # Capitalizar palavras
            
            # Verifica se o diagnóstico predito começa com "Risco"
            if diagnostico.startswith("Risco"):
                mensagem = f"Diagnóstico: {formatar_diagnostico(diagnostico)}. Fator Relacionado: {sintoma_selecionado}"
            else:
                mensagem = f"Diagnóstico: {formatar_diagnostico(diagnostico)}. Característica Definidora: {sintoma_selecionado}"
                
            escolhido = st.checkbox(mensagem)
            
            if escolhido:
                if diagnostico not in diagnosticos_selecionados:
                    diagnosticos_selecionados[diagnostico] = []
                diagnosticos_selecionados[diagnostico].append(sintoma)


    diagnostico_personalizado = st.text_area("Se desejar, escreva um diagnóstico personalizado:", "")

    if diagnostico_personalizado:
        diagnosticos_selecionados["Personalizado"] = [diagnostico_personalizado]

    if diagnosticos_selecionados:
        st.subheader("Diagnósticos Selecionados:")

        for diagnostico, sintomas_selecionados in diagnosticos_selecionados.items():
            sintomas_formatados = [formatar_diagnostico(sintoma) for sintoma in sintomas_selecionados]

            sintomas_concatenados = ", ".join(sintomas_formatados)

            # Verifica se o diagnóstico selecionado começa com "Risco"
            if diagnostico.startswith("Risco"):
                mensagem = f"<span style='font-size:18px'>Diagnóstico: {formatar_diagnostico(diagnostico)}. Fatores Relacionados: {sintomas_concatenados}</span>"
            else:
                mensagem = f"<span style='font-size:18px'>Diagnóstico: {formatar_diagnostico(diagnostico)}. Características Definidoras: {sintomas_concatenados}</span>"

            st.markdown(mensagem, unsafe_allow_html=True)

            if diagnostico in cuidados_df.columns:
                cuidados_relacionados = cuidados_df[diagnostico].iloc[0]
                if cuidados_relacionados:
                    st.write("")
                    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;**Cuidados Relacionados:**")
                    for cuidado in cuidados_relacionados.split('\t'):
                        st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• " + cuidado, unsafe_allow_html=True)


    observacoes = st.text_area("Observações gerais sobre os diagnósticos selecionados:", "")
    avaliacao = st.slider("Avalie a aplicação (de 1 a 5):", 1, 5)

    if st.button("Salvar Avaliação"):
        # Registro do tempo de fim da seleção de sintomas
        fim_secao = time.time()
        tempo_decorrido = round(fim_secao - inicio_secao, 2)
        salvar_arquivo(iniciais_usuario, diagnosticos_selecionados, sintomas, observacoes, avaliacao, tempo_decorrido)


def salvar_arquivo(iniciais_usuario, diagnosticos_selecionados, sintomas, observacoes, avaliacao, tempo_decorrido):
    try:
        with open("resultado_avaliacoes.csv", mode="a", newline="") as file:
            writer = csv.writer(file)

            if file.tell() == 0:
                writer.writerow(['Iniciais do Usuário', 'Diagnóstico', 'Sintomas', 'Observações', 'Avaliação', 'Tempo de Seleção (s)'])

            for diagnostico, sintomas_diagnostico in diagnosticos_selecionados.items():
                sintomas_formatados = [formatar_diagnostico(sintoma) for sintoma in sintomas_diagnostico]
                writer.writerow([iniciais_usuario, diagnostico, ", ".join(sintomas_formatados), observacoes, avaliacao, tempo_decorrido])
        st.success("Avaliação salva com sucesso.")
    except Exception as e:
        st.error(f"Ocorreu um erro ao salvar os dados: {str(e)}")


if __name__ == "__main__":
    main()

    main()
