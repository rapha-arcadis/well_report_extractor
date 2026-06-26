import json
import os
import pandas as pd
from typing import Dict, List

from src.llm import OpenAILLM


# Defining the root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class ReportV2Model:
    def __init__(self):
        self.img_path = os.path.join(ROOT_DIR, r"well_v2.png")

        self.get_monitoring_well_data()

    def get_monitoring_well_data(self) -> List[Dict]:
        # Read prompt from file
        prompt_path = os.path.join(ROOT_DIR, r'resources\prompts\perfil_v2\perfil_dados\extrai_dados_perfil.txt')

        with open(prompt_path, 'r') as f:
            prompt_str = f.read()

        llm_result = OpenAILLM().chat_completion(
            messages=[{"role": "user", "content": prompt_str}],
            image_path=self.img_path
        )

        # Deleting result filler
        llm_result = llm_result.replace('```json\n', '')
        llm_result = llm_result.replace('\n```', '')

        # Formatting llm_result
        llm_result = json.loads(llm_result)

        well_df = pd.DataFrame(llm_result['well_data'])

        well_records = well_df.to_dict('records')

        return well_records


class ReportV1Model:
    def __init__(self):
        self.img_path = os.path.join(ROOT_DIR, r"well_v1.png")

        self.get_monitoring_well_data()

    def get_monitoring_well_data(self) -> List[Dict]:
        # Read prompt from file
        prompt_path = os.path.join(ROOT_DIR, r'resources\prompts\perfil_v2\perfil_dados\extrai_dados_perfil.txt')

        with open(prompt_path, 'r') as f:
            prompt_str = f.read()

        llm_result = OpenAILLM().chat_completion(
            messages=[{"role": "user", "content": prompt_str}],
            image_path=self.img_path
        )

        # Deleting result filler
        llm_result = llm_result.replace('```json\n', '')
        llm_result = llm_result.replace('\n```', '')

        # Formatting llm_result
        llm_result = json.loads(llm_result)

        well_df = pd.DataFrame(llm_result['well_data'])
        well_df['Observacao'] = [''] * len(well_df.index)
        for depth, val in llm_result['obs'].items():
            well_df['Observacao'] = well_df.apply(
                lambda x:
                    f"{val}; {x['Observacao']}"
                    if (x['Profundidade_inicial_instalacao'] < float(depth) <= x['Profundidade_final_instalacao'])
                    else x['Observacao'],
                axis=1)

        well_records = well_df.to_dict('records')

        return well_records


if __name__ == '__main__':
    ReportV2Model()
