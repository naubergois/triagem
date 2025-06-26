"""Agente de análise médica usando LangChain e OpenAI."""

import json
import os
from dataclasses import dataclass
from typing import Dict

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI


load_dotenv()


@dataclass
class PatientAnalysisAgent:
    """Agente que gera avaliação clínica e pontuação de gravidade."""

    model_name: str = "gpt-3.5-turbo"

    def __post_init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY não definido no .env")
        self.llm = ChatOpenAI(model=self.model_name, openai_api_key=api_key)
        template = (
            "Você é um médico especialista em triagem."
            " Analise os dados do paciente abaixo e dê uma breve avaliação clínica."
            " Em seguida, atribua uma pontuação geral de gravidade entre 0 e 10."
            "\nDados do paciente:\n{patient_info}\n\n"
            "Responda no formato:\n"
            "Avaliação: <texto>\nPontuação: <número>"
        )
        prompt = PromptTemplate.from_template(template)
        self.chain = LLMChain(llm=self.llm, prompt=prompt)

    def evaluate(self, patient_data: Dict) -> str:
        patient_json = json.dumps(patient_data, indent=2, ensure_ascii=False)
        return self.chain.run(patient_info=patient_json)


def main() -> None:
    import pandas as pd

    df = pd.read_csv("data/patients_data.csv")
    patient = df.iloc[0].to_dict()
    agent = PatientAnalysisAgent()
    result = agent.evaluate(patient)
    print(result)


if __name__ == "__main__":
    main()
