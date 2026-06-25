import io
import re
import fitz
import easyocr
import numpy as np
import pandas as pd
from typing import Dict, List

from PIL import Image


class ExtractData:
    """
    Classe responsável por extrair dados estruturados de relatórios de poços em PDF da WALM
    """

    def __init__(self, pdf_path: str, well_records: List[Dict]) -> None:
        """
        Inicializa a classe com os caminhos dos arquivos de entrada e o motor de OCR.
        """
        self.pdf_path = pdf_path
        self.well_records = well_records
        self.reader = easyocr.Reader(["pt", "en"], gpu=False)

    def extract_text_df(self, bloco_ignorado: tuple = None) -> pd.DataFrame:
        """
        Abre o PDF, renderiza as páginas em imagens de alta resolução, aplica OCR para
        extrair o texto bruto e filtra os metadados específicos do poço baseando-se em regras de proximidade.
        """
        # Abre o documento PDF com o PyMuPDF (fitz)
        doc = fitz.open(self.pdf_path)

        rows = []
        indice = 0

        # Itera por todas as páginas do PDF para aplicar o OCR em formato de imagem
        for page_number, page in enumerate(doc, start=1):
            # Renderiza a página como imagem (pixmap) com qualidade de 300 DPI
            pix = page.get_pixmap(dpi=300)

            # Converte os bytes brutos da imagem em um objeto manipulável pela biblioteca PIL
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            img_np = np.array(img)

            # Se uma região restrita foi passada para ser ignorada, mascara-se de branco
            if bloco_ignorado:
                x0, y0, x1, y1 = bloco_ignorado

                # O Numpy acessa as dimensões da imagem no formato: imagem[y0:y1, x0:x1]
                if len(img_np.shape) == 3:
                    img_np[y0:y1, x0:x1] = [255, 255, 255]
                else:
                    img_np[y0:y1, x0:x1] = 255

            # Executa o OCR na imagem aplicando tolerâncias horizontais para evitar quebra de hifens
            results = self.reader.readtext(
                img_np,
                detail=1,
                width_ths=2.0,  # Margem horizontal de tolerância para agrupar palavras na mesma linha
                x_ths=2.0,  # Tolerância do eixo X para manter blocos de texto contínuos
            )

            # Salva o texto bruto recuperado em uma lista sequencial de dicionários
            for bbox, text, confidence in results:
                rows.append(
                    {
                        "texto": text,
                    }
                )
                indice += 1

        # Transforma a lista de textos brutos extraídos em um DataFrame inicial
        df = pd.DataFrame(rows)

        # Tupla com os termos gatilhos que precisamos localizar no relatório
        termos = ("Identificação", "Início:", "Diâmetro do Poço")

        # Encontra os índices exatos de quais linhas contêm ou começam com esses termos
        linha_termo = df["texto"].str.strip().str.startswith(termos, na=False)
        indices = df[linha_termo].index.tolist()

        indices_finais = []

        # Varre os índices encontrados para aplicar as regras de deslocamento de linha
        for idx in indices:
            texto_atual = df.loc[idx, "texto"].strip()

            # Sempre inclui o índice do próprio termo gatilho
            indices_finais.append(idx)

            # Regra para "Identificação": o valor real (Ex: PM-02) está 2 linhas abaixo do título
            if texto_atual.startswith("Identificação"):
                if idx + 2 < len(
                    df
                ):  # Previne estouro de índice se estiver no fim do arquivo
                    indices_finais.append(idx + 2)

            # Regra para "Diâmetro do Poço": verifica se o valor está na mesma linha ou na linha seguinte
            elif texto_atual.startswith("Diâmetro do Poço"):
                # Caso a tolerância horizontal tenha juntado o título e o valor (ex: "Diâmetro do Poço - pol: 2.0")
                match_num = re.search(
                    r"(?:pol:?\s*|:\s*)(\d+(?:[.,]\d+)?)", texto_atual, re.IGNORECASE
                )
                if not match_num:
                    # Se não achou o número na mesma linha, inclui o índice da linha de baixo como valor
                    if idx + 1 < len(df):
                        indices_finais.append(idx + 1)

        # Filtra o DataFrame original trazendo apenas as linhas necessárias e reseta o índice numérico
        df_filtrado = df.iloc[indices_finais].copy().reset_index(drop=True)

        dados_extraidos = {}

        # Loop de processamento para estruturar chave: valor no dicionário final
        for i in range(len(df_filtrado)):
            texto = str(df_filtrado.loc[i, "texto"]).strip()

            # Mapeamento do Nome do Ponto buscando na linha abaixo (i+1 do DataFrame filtrado)
            if texto.startswith("Identificação"):
                chave = "Identificação da Sondagem do Poço"
                valor = (
                    df_filtrado.loc[i + 1, "texto"].strip()
                    if (i + 1) < len(df_filtrado)
                    else ""
                )
                dados_extraidos[chave] = valor

            # Mapeamento do Diâmetro do Poço utilizando regex ou buscando na linha abaixo
            elif texto.startswith("Diâmetro do Poço"):
                chave = "Diâmetro do Poço - pol"
                match_num = re.search(
                    r"(?:pol:?\s*|:\s*)(\d+(?:[.,]\d+)?)", texto, re.IGNORECASE
                )

                if match_num:
                    valor = match_num.group(1).strip()
                else:
                    valor = (
                        df_filtrado.loc[i + 1, "texto"].strip()
                        if (i + 1) < len(df_filtrado)
                        else ""
                    )
                dados_extraidos[chave] = valor

            # Mapeamento das datas utilizando Expressão Regular para separar Início e Término na mesma string
            elif texto.startswith("Início:"):
                match = re.search(
                    r"Início:\s*\|?\s*(.*?)\s*\|?\s*Término:\s*\|?\s*(.*)", texto
                )
                if match:
                    # Limpa os caracteres (|)
                    dados_extraidos["Início"] = match.group(1).replace("|", "").strip()
                    dados_extraidos["Término"] = match.group(2).replace("|", "").strip()

        # Cria o DataFrame final de 1 linha consolidando o dicionário estruturado
        df_final = pd.DataFrame([dados_extraidos])

        # Renomeia as colunas
        df_final = df_final.rename(
            columns={
                "Identificação da Sondagem do Poço": "Nome_ponto",
                "Início": "Data_inicial_instalacao",
                "Término": "Data_final_instalacao",
                "Diâmetro do Poço - pol": "Diametro_poco",
            }
        )
        df_final["Observação"] = ""

        return df_final

    def concat_json(self, df_pdf: pd.DataFrame) -> pd.DataFrame:
        """
        Lê o JSON e une com o Dataframe.
        """
        # Carrega os dados brutos das camadas a partir do arquivo JSON
        df_json = pd.read_json(self.well_records)

        # Para cada coluna existente no PDF, cria uma correspondente no JSON
        # replicando o valor do índice 0 para todas as linhas da tabela
        for column in df_pdf.columns:
            df_json[column] = df_pdf.loc[0, column]

        # Segrega as listas de colunas para organizar o posicionamento à esquerda e à direita
        columns_pdf = list(df_pdf.columns)
        columns_json = [c for c in df_json.columns if c not in columns_pdf]

        # Concatena a estrutura das colunas mesclando ambos os blocos de dados
        df_completo = df_json[columns_pdf + columns_json].copy()

        # Reordena a sequência das colunas
        df_completo = df_completo.reindex(
            columns=[
                "Nome_ponto",
                "Data_inicial_instalacao",
                "Data_final_instalacao",
                "Nivel_agua_poco",
                "Tipo_segmento",
                "Tipo_material",
                "Profundidade_inicial_instalacao",
                "Profundidade_final_instalacao",
                "Unidade_medida_instalacao",
                "Diametro_poco",
                "Observação",
            ]
        )

        return df_completo

    def exportar_para_excel(
        self, bloco_ignorado: tuple = None, output_path: str = "tabela_final.xlsx"
    ) -> pd.DataFrame:
        """
        Método orquestrador da classe.
        Extração do PDF -> Leitura e União com JSON -> Redimensionamento e Salvamento em Planilha Excel.
        """
        # Executa o pipeline sequencialmente
        df_pdf = self.extract_text_df(bloco_ignorado=bloco_ignorado)
        df_final_completo = self.concat_json(df_pdf)

        # Inicia a gravação do arquivo Excel utilizando a engine openpyxl através do gerenciador de contexto (with)
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_final_completo.to_excel(
                writer, index=False, sheet_name="Dados Consolidados"
            )

            # Acessa a aba criada para configurar parâmetros visuais e larguras de coluna
            worksheet = writer.sheets["Dados Consolidados"]
            for idx, col in enumerate(df_final_completo.columns, 1):

                # Calcula dinamicamente o comprimento ideal avaliando o maior texto presente na coluna
                max_len = (
                    max(df_final_completo[col].astype(str).map(len).max(), len(col)) + 2
                )  # Adiciona uma margem de segurança de 2 caracteres

                col_letter = chr(
                    64 + idx
                )  # Mapeamento numérico simples para letras do Excel (1=A, 2=B, etc)
                worksheet.column_dimensions[col_letter].width = min(
                    max_len, 50
                )  # Limita em no máximo 50 de largura

        return df_final_completo


# Para testar
#
# pdf_path = "Docs/22062026_modelo_perfil_v1.pdf"
# json_path = "Docs/dados.json"
#
# fun = ExtractData(pdf_path, json_path)
# fun.exportar_para_excel(
#     bloco_ignorado=(177, 662, 1504, 2653), output_path="Output/resultado_final.xlsx"
# )
