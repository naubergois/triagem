import json
from dataclasses import dataclass

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import FakeListLLM

from models.diabetes_risk import DiabetesRiskModel, age_range_to_value


@dataclass
class DiabetesRiskTools:
    model: DiabetesRiskModel

    def compute_diabetes_score(self, patient_json: str, run_manager=None) -> str:
        data = json.loads(patient_json)
        age = age_range_to_value(data.get("age_range", "31-45"))
        bp = data.get("vital_signs", {}).get("blood_pressure_systolic", 120)
        prob = self.model.predict_proba(age, bp)
        return json.dumps({"score": prob})

    def explain_score(self, score_json: str, run_manager=None) -> str:
        score = json.loads(score_json).get("score", 0)
        if score >= 0.5:
            return f"Alta probabilidade de diabetes ({score:.2f})."
        return f"Baixa probabilidade de diabetes ({score:.2f})."


class DiabetesRiskAgent:
    def __init__(self, model: DiabetesRiskModel):
        self.tools_handler = DiabetesRiskTools(model)
        self.tools = self._create_tools()
        self.llm = FakeListLLM(
            responses=[
                "Vou calcular o risco de diabetes do paciente.",
                "Analisando a pontuação para gerar a explicação.",
            ]
        )
        self.agent = self._create_agent()

    def _create_tools(self):
        return [
            Tool(
                name="compute_diabetes_score",
                description="Calcula a probabilidade de diabetes do paciente",
                func=self.tools_handler.compute_diabetes_score,
            ),
            Tool(
                name="explain_score",
                description="Fornece explicação sobre a pontuação de diabetes",
                func=self.tools_handler.explain_score,
            ),
        ]

    def _create_agent(self):
        prompt = PromptTemplate.from_template(
            """
Você é um especialista em diabetes. Utilize as ferramentas disponíveis para calcular a pontuação de um paciente e explique o resultado.

Pergunta: {input}
Pensamento: {agent_scratchpad}
"""
        )
        return create_react_agent(self.llm, self.tools, prompt)

    def get_explanation(self, patient_data: dict) -> str:
        executor = AgentExecutor(
            agent=self.agent, tools=self.tools, verbose=False, max_iterations=5
        )
        response = executor.invoke({"input": json.dumps(patient_data)})
        return response["output"]

