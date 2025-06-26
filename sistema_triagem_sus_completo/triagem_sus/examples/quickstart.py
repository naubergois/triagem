"""Quickstart example demonstrating patient analysis in a single script."""

import os
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI


@dataclass
class PatientAnalysisAgent:
    """Simple agent that returns a clinical assessment and severity score."""

    model_name: str = "gpt-3.5-turbo"

    def __post_init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.llm = ChatOpenAI(model=self.model_name, openai_api_key=api_key)
        template = (
            "Você é um médico especialista em triagem."
            " Analise os dados do paciente abaixo e dê uma breve avaliação clínica."
            " Em seguida, atribua uma pontuação geral de gravidade entre 0 e 10."
            "\n\nDados do paciente:\n{patient_info}\n\n"
            "Responda no formato:\n"
            "Avaliação: <texto>\nPontuação: <número>"
        )
        prompt = PromptTemplate(input_variables=["patient_info"], template=template)
        self.chain = LLMChain(llm=self.llm, prompt=prompt)

    def evaluate(self, patient_data: Dict) -> str:
        patient_json = json.dumps(patient_data, ensure_ascii=False, indent=2)
        return self.chain.run(patient_info=patient_json)


def prompt_new_patient() -> Dict:
    """Prompt the user to input a new patient's triage data."""
    print("\nInsira os dados da nova triagem:")
    return {
        "nome": input("Nome: "),
        "idade": int(input("Idade: ")),
        "sexo": input("Sexo (Masculino/Feminino): "),
        "sintomas": input("Sintomas (separados por vírgula): "),
        "pressao_arterial": input("Pressão arterial (ex: 120/80): "),
        "frequencia_cardiaca": int(input("Frequência cardíaca: ")),
        "temperatura": float(input("Temperatura: ")),
        "saturacao_oxigenio": int(input("Saturação de oxigênio: ")),
        "historico_medico": input("Histórico médico: "),
    }


def parse_score(result: str) -> float:
    """Extract the numeric score from the agent's response."""
    match = re.search(r"Pontuação:\s*(\d+(?:\.\d+)?)", result)
    return float(match.group(1)) if match else 0.0


def main() -> None:
    """Run patient analysis on a CSV file and optionally add a new triage."""
    path = os.path.join(os.path.dirname(__file__), "example_patients.csv")
    df = pd.read_csv(path)

    agent = PatientAnalysisAgent()

    results: List[Tuple[str, float]] = []
    new_index = None

    answer = input("Deseja adicionar uma nova triagem? (s/n): ").strip().lower()
    if answer == "s":
        patient = prompt_new_patient()
        df.loc[len(df)] = patient
        new_index = len(df) - 1

        # Analyze the newly added triage immediately
        response = agent.evaluate(patient)
        score = parse_score(response)
        results.append((patient["nome"], score))
        print("\nNovo paciente:", patient["nome"])
        print(response)

    for idx, row in df.iterrows():
        if idx == new_index:
            # já analisado acima
            continue
        response = agent.evaluate(row.to_dict())
        score = parse_score(response)
        results.append((row["nome"], score))
        print("\nPaciente:", row["nome"])
        print(response)

    names, scores = zip(*results)
    plt.figure(figsize=(8, 4))
    plt.bar(names, scores, color="skyblue")
    plt.ylabel("Pontuação de Gravidade")
    plt.title("Resultados da Triagem")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "triage_scores.png")
    plt.savefig(output_path)
    print(f"\nGráfico salvo em {output_path}")


if __name__ == "__main__":
    main()
