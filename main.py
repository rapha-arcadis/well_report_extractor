import io
import re
import fitz
import easyocr
import numpy as np
import pandas as pd

from PIL import Image

class ExtractData:
    def __init__(self, pdf_path: str) -> None:

        self.pdf_path = pdf_path
        self.reader = easyocr.Reader(["pt", "en"], gpu=False)
    
    def extract_text_df(self, bloco_ignorado: tuple = None) -> pd.DataFrame:

        doc = fitz.open(self.pdf_path)

        rows = []
        indice = 0

        for page_number, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=300)

            img = Image.open(io.BytesIO(pix.tobytes("png")))
            img_np = np.array(img)

            if bloco_ignorado:
                x0, y0, x1, y1 = bloco_ignorado

                # Pinta a area selecionada de branco
                # O Numpy acessa as dimensões da imagem no formato: imagem[y0:y1, x0:x1]
                if len(img_np.shape) == 3:
                    img_np[y0:y1, x0:x1] = [255, 255, 255]
                else:
                    img_np[y0:y1, x0:x1]
            
            Image.fromarray(img_np).save(f"teste_pagina_{page_number}.png")

            # Passamos parâmetros para forçar a união de blocos na mesma linha
            results = self.reader.readtext(
                img_np, 
                detail=1,
                width_ths=2.0, # Aumenta a tolerância horizontal para juntar os textos
                x_ths=2.0      # Você pode precisar calibrar esse número (1.5, 2.0, 3.0)
            )

            for bbox, text, confidence in results:
                rows.append({
                    "texto": text,
                })

                indice += 1

        df = pd.DataFrame(rows)

        # Tupla com termos desejados
        termos = ("Identificação", "Início:", "Diâmetro do Poço")

        # Encontra quais linhas começam com os termos
        linha_termo = df["texto"].str.strip().str.startswith(termos, na=False)
        indices = df[linha_termo].index.tolist()

        indices_finais = []

        # Varre os indices encontrados para aplicas as regras
        for idx in indices:
            texto_atual = df.loc[idx, "texto"].strip()
            
            indices_finais.append(idx)

            # Regra para "Identificação": trazer a linha índice + 2
            if texto_atual.startswith("Identificação"):
                if idx + 2 < len(df): # Previne erro caso esteja no final do arquivo
                    indices_finais.append(idx + 2)
            
            # Regra para "Diâmetro do Poço": trazer a linha índice + 1
            elif texto_atual.startswith("Diâmetro do Poço"):
                if idx + 1 < len(df):
                    indices_finais.append(idx + 1)
            

        # Filtra o DataFrame original usando a lista de índices processados
        df_filtrado = df.iloc[indices_finais].copy().reset_index(drop=True)

        dados_extraidos = {}

        for i in range(len(df_filtrado)):
            texto = str(df_filtrado.loc[i, "texto"]).strip()

            if texto.startswith("Identificação"):
                chave = "Identificação da Sondagem do Poço"
                valor = df_filtrado.loc[i+1, "texto"].strip() if (i + 1) < len(df_filtrado) else ""
                dados_extraidos[chave] = valor

            elif texto.startswith("Diâmetro do Poço"):
                chave = "Diâmetro do Poço - pol"
                valor = df_filtrado.loc[i+1, "texto"].strip() if (i + 1) < len(df_filtrado) else ""
                dados_extraidos[chave] = valor

            elif texto.startswith("Início:"):
                # Regex para extrair as datas e ignorar os pipes
                match = re.search(r"Início:\s*\|?\s*(.*?)\s*\|?\s*Término:\s*\|?\s*(.*)", texto)
                if match:
                    dados_extraidos["Início"] = match.group(1).replace("|", "").strip()
                    dados_extraidos["Término"] = match.group(2).replace("|", "").strip()

        # Cria o DataFrame final de 1 linha com os dados mapeados
        df_final = pd.DataFrame([dados_extraidos])

        df_final = df_final.rename(columns={
            "Identificação da Sondagem do Poço": "Nome_ponto",
            "Início": "Data_inicial_instalacao",
            "Término": "Data_final_instalacao",
            "Diâmetro do Poço - pol": "Diametro_poco"
        })

        return df_final