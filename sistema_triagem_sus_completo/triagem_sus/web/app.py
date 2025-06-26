"""
Aplicação Web para Sistema de Triagem de Pacientes SUS
Interface web usando Flask para demonstração do sistema
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import sys
import os
import json
import pickle
import pandas as pd
from datetime import datetime

# Adicionar o diretório pai ao path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.intelligent_triage_agent import IntelligentTriageAgent

app = Flask(__name__)
app.secret_key = 'triagem_sus_secret_key'

# Inicializar agente de triagem
triage_agent = None

def load_triage_agent():
    """Carrega o agente de triagem"""
    global triage_agent
    try:
        triage_agent = IntelligentTriageAgent()
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

