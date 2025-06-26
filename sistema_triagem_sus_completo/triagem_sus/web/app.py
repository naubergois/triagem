"""
Aplicação Web para Sistema de Triagem de Pacientes SUS
Interface web usando Flask para demonstração do sistema
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import sys
import os
from io import StringIO
from contextlib import redirect_stdout

# Adicionar o diretório pai ao path para importar módulos antes dos imports locais
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import pickle
import pandas as pd
from sklearn.datasets import load_diabetes
from datetime import datetime
from typing import List, Dict
from heapq import heappush, heappop

from models.diabetes_risk import DiabetesRiskModel, age_range_to_value
from models.pneumonia_detector import PneumoniaXRayModel
from agents.diabetes_risk_agent import DiabetesRiskAgent

from agents.intelligent_triage_agent import IntelligentTriageAgent

app = Flask(__name__)
app.secret_key = 'triagem_sus_secret_key'

# Inicializar agentes e modelo
triage_agent = None
pneumonia_model = PneumoniaXRayModel()
diabetes_model = DiabetesRiskModel()
diabetes_agent = None

# Fila de prioridade (heap)
patient_queue: List[Dict] = []

def load_triage_agent():
    """Carrega agentes e modelos de ML"""
    global triage_agent, diabetes_agent, diabetes_model, pneumonia_model
    try:
        triage_agent = IntelligentTriageAgent()

        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'diabetes_model.pkl')
        if os.path.exists(model_path):
            diabetes_model.load(model_path)
        else:
            diabetes_model.train()
            diabetes_model.save(model_path)

        pneumo_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'pneumonia_model.pt')
        if os.path.exists(pneumo_path):
            pneumonia_model.load(pneumo_path)

        diabetes_agent = DiabetesRiskAgent(diabetes_model)
        return True
    except Exception as e:
        print(f"Erro ao carregar agente: {e}")
        return False

@app.route('/')
def index():
    """Página inicial"""
    return render_template('index.html')

@app.route('/triagem')
def triagem_form():
    """Formulário de triagem"""
    return render_template('triagem.html')

@app.route('/realizar_triagem', methods=['POST'])
def realizar_triagem():
    """Realiza triagem de um paciente"""
    try:
        # Obter dados do formulário
        patient_data = {
            'patient_id': request.form.get('patient_id', ''),
            'age_range': request.form.get('age_range', ''),
            'gender': request.form.get('gender', ''),
            'symptoms': request.form.getlist('symptoms'),
            'comorbidities': request.form.getlist('comorbidities'),
            'chief_complaint': request.form.get('chief_complaint', ''),
            'vital_signs': {
                'temperature': float(request.form.get('temperature', 36.5)),
                'heart_rate': int(request.form.get('heart_rate', 70)),
                'blood_pressure_systolic': int(request.form.get('bp_systolic', 120)),
                'blood_pressure_diastolic': int(request.form.get('bp_diastolic', 80)),
                'respiratory_rate': int(request.form.get('respiratory_rate', 16)),
                'oxygen_saturation': int(request.form.get('oxygen_saturation', 98))
            }
        }
        
        # Simular triagem (em um sistema real, usaria o agente)
        result = simulate_triage(patient_data)

        # Calcular probabilidade de diabetes
        age_val = age_range_to_value(patient_data['age_range'])
        bp_val = patient_data['vital_signs']['blood_pressure_systolic']
        diabetes_score = diabetes_model.predict_proba(age_val, bp_val)

        # Adicionar à fila de prioridade
        heappush(patient_queue, (result['priority'], {
            'patient_id': patient_data['patient_id'],
            'triage_result': result,
            'diabetes_score': diabetes_score,
            'data': patient_data
        }))

        return render_template('resultado.html', result=result, patient_data=patient_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def simulate_triage(patient_data):
    """Simula triagem baseada nos dados do paciente"""
    # Lógica simplificada de triagem
    risk_score = 0
    
    # Avaliar sintomas
    high_risk_symptoms = ['dor torácica', 'dificuldade respiratória', 'inconsciência']
    moderate_risk_symptoms = ['febre alta', 'vômitos', 'dor abdominal']
    
    for symptom in patient_data['symptoms']:
        if any(hrs in symptom.lower() for hrs in high_risk_symptoms):
            risk_score += 3
        elif any(mrs in symptom.lower() for mrs in moderate_risk_symptoms):
            risk_score += 2
        else:
            risk_score += 1
    
    # Avaliar sinais vitais
    vitals = patient_data['vital_signs']
    
    if vitals['temperature'] > 39 or vitals['temperature'] < 35:
        risk_score += 2
    if vitals['heart_rate'] > 120 or vitals['heart_rate'] < 50:
        risk_score += 2
    if vitals['blood_pressure_systolic'] > 180 or vitals['blood_pressure_systolic'] < 90:
        risk_score += 2
    if vitals['oxygen_saturation'] < 95:
        risk_score += 3
    
    # Determinar categoria de risco
    if risk_score >= 8:
        risk_category = 'VERMELHO'
        priority = 1
        max_wait = 0
    elif risk_score >= 6:
        risk_category = 'LARANJA'
        priority = 2
        max_wait = 10
    elif risk_score >= 4:
        risk_category = 'AMARELO'
        priority = 3
        max_wait = 60
    elif risk_score >= 2:
        risk_category = 'VERDE'
        priority = 4
        max_wait = 120
    else:
        risk_category = 'AZUL'
        priority = 5
        max_wait = 240
    
    # Gerar recomendações
    recommendations = {
        'VERMELHO': ['Atendimento imediato', 'Monitorização contínua', 'Preparar ressuscitação'],
        'LARANJA': ['Atendimento em 10 min', 'Monitorizar sinais vitais', 'Acesso venoso'],
        'AMARELO': ['Atendimento em 1 hora', 'Reavaliação em 30 min', 'Observação'],
        'VERDE': ['Atendimento em 2 horas', 'Orientações gerais', 'Reavaliação se necessário'],
        'AZUL': ['Atendimento em 4 horas', 'Consulta de rotina', 'Orientações preventivas']
    }
    
    return {
        'risk_category': risk_category,
        'priority': priority,
        'max_wait_minutes': max_wait,
        'confidence_score': min(0.95, 0.6 + (risk_score * 0.05)),
        'reasoning': f'Score de risco calculado: {risk_score}',
        'recommendations': recommendations[risk_category],
        'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    }

@app.route('/dashboard')
def dashboard():
    """Dashboard com estatísticas"""
    try:
        # Carregar dados de triagem
        with open('/home/ubuntu/triagem_sus/data/triage_results.json', 'r') as f:
            results = json.load(f)
        
        # Calcular estatísticas
        stats = {
            'total_patients': len(results),
            'risk_distribution': {},
            'avg_confidence': 0
        }
        
        for result in results:
            risk = result['risk_category']
            stats['risk_distribution'][risk] = stats['risk_distribution'].get(risk, 0) + 1
            stats['avg_confidence'] += result['confidence_score']
        
        stats['avg_confidence'] /= len(results) if results else 1

        return render_template('dashboard.html', stats=stats, results=results)

    except FileNotFoundError:
        return render_template('dashboard.html', stats={}, results=[])


@app.route('/dados_diabetes')
def dados_diabetes():
    """Exibe uma amostra do dataset de diabetes"""
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    return render_template('dados_diabetes.html', columns=df.columns[:4], rows=df.head().iloc[:, :4].values)


@app.route('/retrain')
def retrain_models():
    return render_template('retrain.html')


@app.route('/train/diabetes')
def train_diabetes():
    buffer = StringIO()
    with redirect_stdout(buffer):
        acc = diabetes_model.train()
        path = os.path.join(os.path.dirname(__file__), '..', 'models', 'diabetes_model.pkl')
        diabetes_model.save(path)
        print(f"Acurácia final: {acc:.4f}")
    logs = buffer.getvalue().splitlines()
    return render_template('train_result.html', logs=logs, accuracy=acc, model='Diabetes')


@app.route('/train/pneumonia')
def train_pneumonia():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'chest_xray')
    buffer = StringIO()
    try:
        with redirect_stdout(buffer):
            acc = pneumonia_model.train(data_dir)
            path = os.path.join(os.path.dirname(__file__), '..', 'models', 'pneumonia_model.pt')
            pneumonia_model.save(path)
            print(f"Acurácia final: {acc:.4f}")
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 500

    logs = buffer.getvalue().splitlines()
    return render_template('train_result.html', logs=logs, accuracy=acc, model='Pneumonia')


@app.route('/fila')
def fila_prioridade():
    """Exibe a fila de pacientes ordenada por prioridade"""
    fila = [item[1] for item in sorted(patient_queue, key=lambda x: x[0])]
    return render_template('fila.html', fila=fila)


@app.route('/explicacao/<patient_id>')
def explicacao_paciente(patient_id: str):
    """Gera explicação do risco de diabetes para um paciente da fila"""
    for _, item in patient_queue:
        if item['patient_id'] == patient_id:
            texto = diabetes_agent.get_explanation(item['data'])
            return jsonify({'explicacao': texto, 'score': item['diabetes_score']})
    return jsonify({'error': 'Paciente não encontrado'}), 404

@app.route('/api/triagem', methods=['POST'])
def api_triagem():
    """API endpoint para triagem"""
    try:
        data = request.get_json()
        result = simulate_triage(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Iniciando aplicação web...")
    load_triage_agent()
    app.run(host='0.0.0.0', port=5000, debug=True)

