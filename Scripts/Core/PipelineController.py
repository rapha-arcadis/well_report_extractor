import fitz
import cv2
import numpy as np
import pandas as pd
from typing import Dict

from Scripts.ExtractData import ExtractData
from Scripts.ReportModel import ReportV1Model


class PipelineController:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def processar_relatorio(self) -> pd.DataFrame:
        # Abre o PDF e renderiza a primeira página a 300 DPI
        doc = fitz.open(self.pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=300)

        # Converte a imagem renderizada para o formato Numpy/OpenCV (em memória)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Identifica a região dinamicamente
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        max_area = 0
        best_box = None
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area > max_area:
                max_area = area
                best_box = (x, y, x + w, y + h)  # (x0, y0, x1, y1)

        doc.close()

        if not best_box:
            raise ValueError(
                "Não foi possível identificar a região de interesse no PDF."
            )

        x0, y0, x1, y1 = best_box

        # Recorta a imagem usando as coordenadas e converte para Bytes (memória)
        cropped_img = img_bgr[y0:y1, x0:x1]
        _, buffer = cv2.imencode(".png", cropped_img)
        img_bytes = buffer.tobytes()

        # Envia a imagem (em bytes) para o ReportModel analisar
        report_model = ReportV1Model()
        json_dinamico = report_model.get_monitoring_well_data(image_bytes=img_bytes)

        # Repassa o JSON em memória e a bounding box para o ExtractData
        extractor = ExtractData(pdf_path=self.pdf_path, json_data=json_dinamico)
        df_final = extractor.extract_text_df(bloco_ignorado=best_box)

        return df_final
