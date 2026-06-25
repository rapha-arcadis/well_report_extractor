import json
import base64
import pandas as pd
from typing import Dict, List

from src.llm import OpenAILLM


class ReportV1Model:
    def __init__(self):
        pass

    def get_monitoring_well_data(self, image_bytes: bytes) -> Dict:
        # Read prompt from file
        prompt_path = r"C:\Users\matheus.carneiro\Projetos\well-report-extractor\resources\prompts\perfil_v1\perfil_dados\extrai_dados_perfil.txt"

        with open(prompt_path, "r") as f:
            prompt_str = f.read()

        # Converte os bytes da imagem em Base64 para envio via API
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        llm_result = OpenAILLM().chat_completion(
            messages=[{"role": "user", "content": prompt_str}],
            image_base64=base64_image,
        )

        # Deleting result filler
        llm_result = llm_result.replace("```json\n", "").replace("\n```", "")

        # Formatting llm_result
        llm_result_dict = json.loads(llm_result)

        well_df = pd.DataFrame(llm_result["well_data"])
        well_df["Observacao"] = [""] * len(well_df.index)
        for depth, val in llm_result["obs"].items():
            well_df["Observacao"] = well_df.apply(
                lambda x: (
                    f"{val}; {x['Observacao']}"
                    if (
                        x["Profundidade_inicial_instalacao"]
                        < float(depth)
                        <= x["Profundidade_final_instalacao"]
                    )
                    else x["Observacao"]
                ),
                axis=1,
            )

        # well_records = well_df.to_dict("records")

        return llm_result_dict


# if __name__ == "__main__":
#     ReportV1Model()

# controller = PipelineController(pdf_path="caminho_pdf.pdf")
# planilha_final_df = controller.processar_relatorio()

# # planilha_final_df.to_excel('resultado_final.xlsx', index=False)
