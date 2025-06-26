"""
Agente Inteligente de Triagem de Pacientes SUS
Implementado com LangChain para análise automática e classificação de risco
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_community.llms import FakeListLLM

from agents.patient_analysis_agent import PatientAnalysisAgent

import warnings
warnings.filterwarnings('ignore')

@dataclass
class PatientData:
    """Estrutura de dados do paciente"""
    patient_id: str
    age_range: str
    gender: str
    symptoms: List[str]
    comorbidities: List[str]
    vital_signs: Dict[str, float]
    chief_complaint: str
    arrival_time: str

@dataclass
class TriageResult:
    """Resultado da triagem"""
    patient_id: str
    risk_category: str
    priority: int
    max_wait_minutes: int
    confidence_score: float
    reasoning: str
    recommendations: List[str]
    alerts: List[str]
    timestamp: str

class ManchesterProtocolEngine:
    """Engine do Protocolo de Manchester para classificação de risco"""
    
    def __init__(self):
        self.risk_categories = {
            'VERMELHO': {'priority': 1, 'max_wait_minutes': 0},
            'LARANJA': {'priority': 2, 'max_wait_minutes': 10},
            'AMARELO': {'priority': 3, 'max_wait_minutes': 60},
            'VERDE': {'priority': 4, 'max_wait_minutes': 120},
            'AZUL': {'priority': 5, 'max_wait_minutes': 240}
        }
        
        # Regras críticas para classificação VERMELHO
        self.critical_symptoms = [
            'parada cardiorrespiratória', 'inconsciência', 'convulsões ativas',
            'hemorragia grave', 'choque', 'dificuldade respiratória severa'
        ]
        
        # Regras para sinais vitais críticos
        self.critical_vitals = {
            'temperature': {'min': 35.0, 'max': 41.0},
            'heart_rate': {'min': 50, 'max': 150},
            'blood_pressure_systolic': {'min': 90, 'max': 200},
            'respiratory_rate': {'min': 12, 'max': 25},
            'oxygen_saturation': {'min': 95, 'max': 100}
        }
    
    def classify_risk(self, patient: PatientData) -> Dict[str, Any]:
        """Classifica o risco do paciente baseado no protocolo de Manchester"""
        
        # Verificar sintomas críticos
        for symptom in patient.symptoms:
            if any(critical in symptom.lower() for critical in self.critical_symptoms):
                return {
                    'risk_category': 'VERMELHO',
                    'reasoning': f'Sintoma crítico identificado: {symptom}',
                    'confidence': 0.95
                }
        
        # Verificar sinais vitais críticos
        vital_alerts = []
        for vital, value in patient.vital_signs.items():
            if vital in self.critical_vitals:
                limits = self.critical_vitals[vital]
                if value < limits['min'] or value > limits['max']:
                    vital_alerts.append(f'{vital}: {value}')
        
        if len(vital_alerts) >= 2:
            return {
                'risk_category': 'VERMELHO',
                'reasoning': f'Múltiplos sinais vitais alterados: {", ".join(vital_alerts)}',
                'confidence': 0.90
            }
        elif len(vital_alerts) == 1:
            return {
                'risk_category': 'LARANJA',
                'reasoning': f'Sinal vital alterado: {vital_alerts[0]}',
                'confidence': 0.80
            }
        
        # Classificação baseada em sintomas
        high_risk_symptoms = ['dor torácica', 'dificuldade respiratória', 'febre alta']
        moderate_risk_symptoms = ['dor abdominal', 'vômitos', 'cefaleia']
        
        for symptom in patient.symptoms:
            if any(high_risk in symptom.lower() for high_risk in high_risk_symptoms):
                return {
                    'risk_category': 'AMARELO',
                    'reasoning': f'Sintoma de risco moderado: {symptom}',
                    'confidence': 0.70
                }
        
        for symptom in patient.symptoms:
            if any(moderate in symptom.lower() for moderate in moderate_risk_symptoms):
                return {
                    'risk_category': 'VERDE',
                    'reasoning': f'Sintoma de baixo risco: {symptom}',
                    'confidence': 0.60
                }
        
        # Caso padrão
        return {
            'risk_category': 'AZUL',
            'reasoning': 'Sintomas não urgentes ou consulta de rotina',
            'confidence': 0.50
        }

class TriageTools:
    """Ferramentas para o agente de triagem"""
    
    def __init__(self):
        self.manchester_engine = ManchesterProtocolEngine()
        self.patient_database = self._load_patient_data()
    
    def _load_patient_data(self) -> pd.DataFrame:
        """Carrega dados dos pacientes"""
        try:
            return pd.read_csv('/home/ubuntu/triagem_sus/data/patients_data.csv')
        except FileNotFoundError:
            return pd.DataFrame()
    
    def get_patient_data(self, patient_id: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Ferramenta para buscar dados do paciente"""
        if self.patient_database.empty:
            return "Base de dados não encontrada"
        
        patient_row = self.patient_database[self.patient_database['patient_id'] == patient_id]
        if patient_row.empty:
            return f"Paciente {patient_id} não encontrado"
        
        patient_data = patient_row.iloc[0].to_dict()
        return json.dumps(patient_data, indent=2, default=str)
    
    def classify_manchester_risk(self, patient_data_json: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Ferramenta para classificar risco usando protocolo de Manchester"""
        try:
            patient_dict = json.loads(patient_data_json)
            
            # Converter para PatientData
            patient = PatientData(
                patient_id=patient_dict['patient_id'],
                age_range=patient_dict['age_range'],
                gender=patient_dict['gender'],
                symptoms=eval(patient_dict['symptoms']) if isinstance(patient_dict['symptoms'], str) else patient_dict['symptoms'],
                comorbidities=eval(patient_dict['comorbidities']) if isinstance(patient_dict['comorbidities'], str) else patient_dict['comorbidities'],
                vital_signs=eval(patient_dict['vital_signs']) if isinstance(patient_dict['vital_signs'], str) else patient_dict['vital_signs'],
                chief_complaint=patient_dict['chief_complaint'],
                arrival_time=patient_dict['arrival_time']
            )
            
            result = self.manchester_engine.classify_risk(patient)
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Erro na classificação: {str(e)}"
    
    def generate_recommendations(self, risk_category: str, symptoms: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Ferramenta para gerar recomendações baseadas no risco"""
        recommendations = {
            'VERMELHO': [
                'Atendimento imediato obrigatório',
                'Monitorização contínua de sinais vitais',
                'Preparar para possível ressuscitação',
                'Comunicar médico emergencista imediatamente'
            ],
            'LARANJA': [
                'Atendimento em até 10 minutos',
                'Monitorização de sinais vitais a cada 15 minutos',
                'Preparar acesso venoso',
                'Comunicar médico plantonista'
            ],
            'AMARELO': [
                'Atendimento em até 1 hora',
                'Reavaliação a cada 30 minutos',
                'Manter paciente em observação',
                'Orientar sobre sinais de alarme'
            ],
            'VERDE': [
                'Atendimento em até 2 horas',
                'Reavaliação se necessário',
                'Orientações gerais de cuidado',
                'Possibilidade de alta com orientações'
            ],
            'AZUL': [
                'Atendimento em até 4 horas',
                'Consulta de rotina',
                'Orientações preventivas',
                'Agendamento de retorno se necessário'
            ]
        }
        
        return json.dumps(recommendations.get(risk_category, []), indent=2)
    
    def check_alerts(self, patient_data_json: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Ferramenta para verificar alertas especiais"""
        try:
            patient_dict = json.loads(patient_data_json)
            alerts = []
            
            # Verificar idade avançada
            if patient_dict['age_range'] in ['61-75', '76+']:
                alerts.append('ATENÇÃO: Paciente idoso - maior risco de complicações')
            
            # Verificar comorbidades
            comorbidities = eval(patient_dict['comorbidities']) if isinstance(patient_dict['comorbidities'], str) else patient_dict['comorbidities']
            high_risk_comorbidities = ['diabetes', 'cardiopatia', 'dpoc']
            
            for comorbidity in comorbidities:
                if comorbidity in high_risk_comorbidities:
                    alerts.append(f'ATENÇÃO: Comorbidade de alto risco - {comorbidity}')
            
            # Verificar sinais vitais
            vital_signs = eval(patient_dict['vital_signs']) if isinstance(patient_dict['vital_signs'], str) else patient_dict['vital_signs']
            
            if vital_signs.get('oxygen_saturation', 100) < 95:
                alerts.append('ALERTA: Saturação de oxigênio baixa')
            
            if vital_signs.get('temperature', 36.5) > 39:
                alerts.append('ALERTA: Febre alta')
            
            return json.dumps(alerts, indent=2)
            
        except Exception as e:
            return f"Erro na verificação de alertas: {str(e)}"

class IntelligentTriageAgent:
    """Agente Inteligente de Triagem usando LangChain"""
    
    def __init__(self):
        self.tools_handler = TriageTools()
        self.tools = self._create_tools()
        self.llm = self._create_llm()
        self.agent = self._create_agent()
        # Agente de análise clínica via OpenAI
        self.analysis_agent = PatientAnalysisAgent()
    
    def _create_llm(self):
        """Cria um LLM simulado para demonstração"""
        responses = [
            "Vou analisar os dados do paciente e classificar o risco conforme o protocolo de Manchester.",
            "Baseado nos sintomas e sinais vitais, vou determinar a prioridade de atendimento.",
            "Analisando as informações clínicas para gerar recomendações apropriadas.",
            "Verificando alertas especiais baseados no perfil do paciente.",
            "Compilando resultado final da triagem com todas as informações relevantes."
        ]
        return FakeListLLM(responses=responses)
    
    def _create_tools(self) -> List[Tool]:
        """Cria as ferramentas do agente"""
        return [
            Tool(
                name="get_patient_data",
                description="Busca dados completos de um paciente pelo ID",
                func=self.tools_handler.get_patient_data
            ),
            Tool(
                name="classify_manchester_risk",
                description="Classifica o risco do paciente usando protocolo de Manchester",
                func=self.tools_handler.classify_manchester_risk
            ),
            Tool(
                name="generate_recommendations",
                description="Gera recomendações baseadas na categoria de risco",
                func=self.tools_handler.generate_recommendations
            ),
            Tool(
                name="check_alerts",
                description="Verifica alertas especiais baseados no perfil do paciente",
                func=self.tools_handler.check_alerts
            )
        ]
    
    def _create_agent(self):
        """Cria o agente ReAct"""
        prompt = PromptTemplate.from_template("""
        Você é um agente inteligente especializado em triagem médica no SUS.
        Sua função é analisar dados de pacientes e classificar o risco conforme o protocolo de Manchester.

        Você tem acesso às seguintes ferramentas:
        {tools}
        
        Use o seguinte formato:
        
        Pergunta: a pergunta de entrada que você deve responder
        Pensamento: você deve sempre pensar sobre o que fazer
        Ação: a ação a tomar, deve ser uma das [{tool_names}]
        Entrada da Ação: a entrada para a ação
        Observação: o resultado da ação
        ... (este Pensamento/Ação/Entrada da Ação/Observação pode repetir N vezes)
        Pensamento: Agora sei a resposta final
        Resposta Final: a resposta final para a pergunta original

        Pergunta: {input}
        Pensamento: {agent_scratchpad}
        """)
        tools_str = "\n".join(f"{t.name}: {t.description}" for t in self.tools)
        tool_names = ", ".join(t.name for t in self.tools)
        prompt = prompt.partial(tools=tools_str, tool_names=tool_names)

        return create_react_agent(self.llm, self.tools, prompt)
    
    def perform_triage(self, patient_id: str) -> TriageResult:
        """Realiza triagem completa de um paciente e gera avaliação via OpenAI"""
        
        # Executar agente
        agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10
        )
        
        try:
            # Buscar dados do paciente
            patient_data = self.tools_handler.get_patient_data(patient_id)
            if "não encontrado" in patient_data:
                raise ValueError(f"Paciente {patient_id} não encontrado")
            
            patient_dict = json.loads(patient_data)
            
            # Classificar risco
            risk_result = self.tools_handler.classify_manchester_risk(patient_data)
            risk_data = json.loads(risk_result)
            
            # Gerar recomendações
            recommendations = self.tools_handler.generate_recommendations(
                risk_data['risk_category'], 
                str(patient_dict['symptoms'])
            )
            
            # Verificar alertas
            alerts = self.tools_handler.check_alerts(patient_data)
            
            # Criar resultado da triagem
            result = TriageResult(
                patient_id=patient_id,
                risk_category=risk_data['risk_category'],
                priority=self.tools_handler.manchester_engine.risk_categories[risk_data['risk_category']]['priority'],
                max_wait_minutes=self.tools_handler.manchester_engine.risk_categories[risk_data['risk_category']]['max_wait_minutes'],
                confidence_score=risk_data['confidence'],
                reasoning=risk_data['reasoning'],
                recommendations=json.loads(recommendations),
                alerts=json.loads(alerts),
                timestamp=datetime.now().isoformat()
            )

            # Avaliação clínica adicional pelo agente OpenAI
            analysis = self.analysis_agent.evaluate(patient_dict)
            print("\nAvaliação do especialista:")
            print(analysis)

            return result
            
        except Exception as e:
            # Resultado de erro
            return TriageResult(
                patient_id=patient_id,
                risk_category='ERRO',
                priority=999,
                max_wait_minutes=0,
                confidence_score=0.0,
                reasoning=f"Erro na triagem: {str(e)}",
                recommendations=[],
                alerts=[f"Erro no sistema: {str(e)}"],
                timestamp=datetime.now().isoformat()
            )
    
    def batch_triage(self, patient_ids: List[str]) -> List[TriageResult]:
        """Realiza triagem em lote"""
        results = []
        for patient_id in patient_ids:
            result = self.perform_triage(patient_id)
            results.append(result)
        return results

def main():
    """Função principal para demonstração"""
    print("=== Sistema de Triagem Inteligente SUS ===")
    print("Inicializando agente inteligente...")
    
    # Criar agente
    agent = IntelligentTriageAgent()
    
    # Testar com alguns pacientes
    test_patients = ['PAC_000001', 'PAC_000002', 'PAC_000003', 'PAC_000004', 'PAC_000005']
    
    print(f"\nRealizando triagem de {len(test_patients)} pacientes...")
    
    results = []
    for patient_id in test_patients:
        print(f"\n--- Triagem do paciente {patient_id} ---")
        result = agent.perform_triage(patient_id)
        results.append(result)
        
        print(f"Categoria de Risco: {result.risk_category}")
        print(f"Prioridade: {result.priority}")
        print(f"Tempo máximo de espera: {result.max_wait_minutes} minutos")
        print(f"Confiança: {result.confidence_score:.2f}")
        print(f"Justificativa: {result.reasoning}")
        
        if result.alerts:
            print(f"Alertas: {', '.join(result.alerts)}")
    
    # Salvar resultados
    results_data = []
    for result in results:
        results_data.append({
            'patient_id': result.patient_id,
            'risk_category': result.risk_category,
            'priority': result.priority,
            'max_wait_minutes': result.max_wait_minutes,
            'confidence_score': result.confidence_score,
            'reasoning': result.reasoning,
            'recommendations': result.recommendations,
            'alerts': result.alerts,
            'timestamp': result.timestamp
        })
    
    # Salvar em JSON
    with open('/home/ubuntu/triagem_sus/data/triage_results.json', 'w') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados salvos em: /home/ubuntu/triagem_sus/data/triage_results.json")
    print("Triagem concluída com sucesso!")

if __name__ == "__main__":
    main()

