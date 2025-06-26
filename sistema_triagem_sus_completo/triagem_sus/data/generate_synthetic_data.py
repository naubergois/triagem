"""
Gerador de dados sintéticos para sistema de triagem de pacientes SUS
Baseado no Protocolo de Manchester e características clínicas reais
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json

# Configuração de seed para reprodutibilidade
np.random.seed(42)
random.seed(42)

class SyntheticPatientDataGenerator:
    def __init__(self):
        # Protocolo de Manchester - Classificação de Risco
        self.manchester_categories = {
            'VERMELHO': {'priority': 1, 'max_wait_minutes': 0, 'description': 'Emergência'},
            'LARANJA': {'priority': 2, 'max_wait_minutes': 10, 'description': 'Muito urgente'},
            'AMARELO': {'priority': 3, 'max_wait_minutes': 60, 'description': 'Urgente'},
            'VERDE': {'priority': 4, 'max_wait_minutes': 120, 'description': 'Pouco urgente'},
            'AZUL': {'priority': 5, 'max_wait_minutes': 240, 'description': 'Não urgente'}
        }
        
        # Sintomas por categoria de risco
        self.symptoms_by_risk = {
            'VERMELHO': [
                'parada cardiorrespiratória', 'inconsciência', 'convulsões ativas',
                'hemorragia grave', 'choque', 'dificuldade respiratória severa',
                'dor torácica com sinais de infarto', 'trauma craniano grave'
            ],
            'LARANJA': [
                'dor torácica intensa', 'dificuldade respiratória moderada',
                'vômitos persistentes', 'febre alta (>39°C)', 'confusão mental',
                'dor abdominal severa', 'cefaleia intensa súbita'
            ],
            'AMARELO': [
                'dor moderada', 'febre (38-39°C)', 'vômitos ocasionais',
                'diarreia', 'tosse persistente', 'dor de cabeça',
                'dor nas costas', 'tontura'
            ],
            'VERDE': [
                'dor leve', 'febre baixa (37-38°C)', 'resfriado',
                'dor de garganta', 'dor muscular leve', 'fadiga',
                'náusea leve', 'constipação'
            ],
            'AZUL': [
                'consulta de rotina', 'renovação de receita',
                'resultado de exame', 'orientação médica',
                'vacinação', 'atestado médico'
            ]
        }
        
        # Dados demográficos
        self.age_ranges = ['0-17', '18-30', '31-45', '46-60', '61-75', '76+']
        self.genders = ['M', 'F']
        self.states = ['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'GO', 'PE', 'CE']
        
        # Comorbidades
        self.comorbidities = [
            'diabetes', 'hipertensão', 'cardiopatia', 'asma', 'dpoc',
            'obesidade', 'depressão', 'ansiedade', 'artrite', 'osteoporose'
        ]
        
    def generate_patient_data(self, n_patients=10000):
        """Gera dados sintéticos de pacientes"""
        patients = []
        
        for i in range(n_patients):
            # Classificação de risco (distribuição realística)
            risk_weights = [0.05, 0.15, 0.30, 0.35, 0.15]  # Vermelho, Laranja, Amarelo, Verde, Azul
            risk_category = np.random.choice(list(self.manchester_categories.keys()), p=risk_weights)
            
            # Dados demográficos
            age_range = np.random.choice(self.age_ranges)
            gender = np.random.choice(self.genders)
            state = np.random.choice(self.states)
            
            # Sintomas baseados na categoria de risco
            num_symptoms = np.random.randint(1, 4)
            symptoms = random.sample(self.symptoms_by_risk[risk_category], 
                                   min(num_symptoms, len(self.symptoms_by_risk[risk_category])))
            
            # Comorbidades (mais prováveis em idades avançadas)
            age_factor = 1 if age_range in ['61-75', '76+'] else 0.3
            num_comorbidities = np.random.poisson(age_factor)
            comorbidities = random.sample(self.comorbidities, 
                                        min(num_comorbidities, len(self.comorbidities)))
            
            # Sinais vitais (simulados baseados na gravidade)
            vital_signs = self._generate_vital_signs(risk_category, age_range)
            
            # Tempo de chegada (últimos 30 dias)
            arrival_time = datetime.now() - timedelta(days=np.random.randint(0, 30),
                                                    hours=np.random.randint(0, 24),
                                                    minutes=np.random.randint(0, 60))
            
            patient = {
                'patient_id': f'PAC_{i+1:06d}',
                'age_range': age_range,
                'gender': gender,
                'state': state,
                'symptoms': symptoms,
                'comorbidities': comorbidities,
                'vital_signs': vital_signs,
                'risk_category': risk_category,
                'priority': self.manchester_categories[risk_category]['priority'],
                'max_wait_minutes': self.manchester_categories[risk_category]['max_wait_minutes'],
                'arrival_time': arrival_time.isoformat(),
                'chief_complaint': self._generate_chief_complaint(symptoms),
                'triage_notes': self._generate_triage_notes(symptoms, vital_signs, risk_category)
            }
            
            patients.append(patient)
        
        return patients
    
    def _generate_vital_signs(self, risk_category, age_range):
        """Gera sinais vitais baseados na categoria de risco"""
        base_vitals = {
            'temperature': 36.5,
            'heart_rate': 70,
            'blood_pressure_systolic': 120,
            'blood_pressure_diastolic': 80,
            'respiratory_rate': 16,
            'oxygen_saturation': 98
        }
        
        # Ajustes baseados na categoria de risco
        risk_adjustments = {
            'VERMELHO': {'temperature': 2, 'heart_rate': 50, 'blood_pressure_systolic': -30, 
                        'respiratory_rate': 10, 'oxygen_saturation': -10},
            'LARANJA': {'temperature': 1.5, 'heart_rate': 30, 'blood_pressure_systolic': -15,
                       'respiratory_rate': 6, 'oxygen_saturation': -5},
            'AMARELO': {'temperature': 1, 'heart_rate': 15, 'blood_pressure_systolic': 10,
                       'respiratory_rate': 3, 'oxygen_saturation': -2},
            'VERDE': {'temperature': 0.5, 'heart_rate': 10, 'blood_pressure_systolic': 5,
                     'respiratory_rate': 2, 'oxygen_saturation': 0},
            'AZUL': {'temperature': 0, 'heart_rate': 0, 'blood_pressure_systolic': 0,
                    'respiratory_rate': 0, 'oxygen_saturation': 0}
        }
        
        vitals = {}
        for vital, base_value in base_vitals.items():
            adjustment = risk_adjustments[risk_category].get(vital, 0)
            noise = np.random.normal(0, abs(adjustment) * 0.2) if adjustment != 0 else np.random.normal(0, 2)
            vitals[vital] = round(base_value + adjustment + noise, 1)
        
        # Garantir valores dentro de limites fisiológicos
        vitals['temperature'] = max(35.0, min(42.0, vitals['temperature']))
        vitals['heart_rate'] = max(40, min(200, int(vitals['heart_rate'])))
        vitals['blood_pressure_systolic'] = max(70, min(250, int(vitals['blood_pressure_systolic'])))
        vitals['blood_pressure_diastolic'] = max(40, min(150, int(vitals['blood_pressure_diastolic'])))
        vitals['respiratory_rate'] = max(8, min(40, int(vitals['respiratory_rate'])))
        vitals['oxygen_saturation'] = max(70, min(100, int(vitals['oxygen_saturation'])))
        
        return vitals
    
    def _generate_chief_complaint(self, symptoms):
        """Gera queixa principal baseada nos sintomas"""
        if not symptoms:
            return "Consulta de rotina"
        
        primary_symptom = symptoms[0]
        complaints = {
            'dor torácica': "Paciente relata dor no peito há algumas horas",
            'febre': "Paciente apresenta febre e mal-estar",
            'dificuldade respiratória': "Paciente com falta de ar e desconforto respiratório",
            'dor abdominal': "Dor na região abdominal com início há algumas horas",
            'cefaleia': "Dor de cabeça intensa",
            'vômitos': "Episódios de vômito",
            'diarreia': "Quadro diarreico",
            'tosse': "Tosse persistente"
        }
        
        for key in complaints:
            if key in primary_symptom:
                return complaints[key]
        
        return f"Paciente apresenta {primary_symptom}"
    
    def _generate_triage_notes(self, symptoms, vital_signs, risk_category):
        """Gera notas de triagem"""
        notes = f"Triagem realizada conforme protocolo de Manchester. "
        notes += f"Classificação: {risk_category}. "
        notes += f"Sintomas: {', '.join(symptoms)}. "
        notes += f"Sinais vitais: PA {vital_signs['blood_pressure_systolic']}/{vital_signs['blood_pressure_diastolic']}, "
        notes += f"FC {vital_signs['heart_rate']}, FR {vital_signs['respiratory_rate']}, "
        notes += f"T {vital_signs['temperature']}°C, SatO2 {vital_signs['oxygen_saturation']}%."
        
        return notes

def main():
    """Função principal para gerar e salvar os dados"""
    generator = SyntheticPatientDataGenerator()
    
    print("Gerando dados sintéticos de pacientes...")
    patients_data = generator.generate_patient_data(n_patients=10000)
    
    # Converter para DataFrame
    df = pd.DataFrame(patients_data)
    
    # Salvar em diferentes formatos
    df.to_csv('/home/ubuntu/triagem_sus/data/patients_data.csv', index=False)
    df.to_json('/home/ubuntu/triagem_sus/data/patients_data.json', orient='records', indent=2)
    
    # Salvar metadados
    metadata = {
        'total_patients': len(patients_data),
        'generation_date': datetime.now().isoformat(),
        'manchester_categories': generator.manchester_categories,
        'risk_distribution': df['risk_category'].value_counts().to_dict()
    }
    
    with open('/home/ubuntu/triagem_sus/data/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dados gerados com sucesso!")
    print(f"Total de pacientes: {len(patients_data)}")
    print(f"Distribuição por categoria de risco:")
    print(df['risk_category'].value_counts())
    
    return df

if __name__ == "__main__":
    df = main()

