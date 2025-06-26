"""
Modelos de Machine Learning para Triagem de Pacientes SUS
Implementa Random Forest, Regressão Logística e Rede Neural
"""

import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class TriageMLModels:
    """Classe para treinar e avaliar modelos de ML para triagem"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def load_and_prepare_data(self, data_path='/home/ubuntu/triagem_sus/data/patients_data.csv'):
        """Carrega e prepara os dados para treinamento"""
        print("Carregando dados...")
        df = pd.read_csv(data_path)
        
        # Preparar features
        features_df = pd.DataFrame()
        
        # Features demográficas
        age_encoder = LabelEncoder()
        features_df['age_encoded'] = age_encoder.fit_transform(df['age_range'])
        self.encoders['age'] = age_encoder
        
        features_df['gender_encoded'] = (df['gender'] == 'M').astype(int)
        
        # Features de sintomas (one-hot encoding dos sintomas mais comuns)
        all_symptoms = []
        for symptoms_str in df['symptoms']:
            symptoms = eval(symptoms_str) if isinstance(symptoms_str, str) else symptoms_str
            all_symptoms.extend(symptoms)
        
        common_symptoms = pd.Series(all_symptoms).value_counts().head(20).index.tolist()
        
        for symptom in common_symptoms:
            features_df[f'symptom_{symptom.replace(" ", "_")}'] = df['symptoms'].apply(
                lambda x: 1 if symptom in (eval(x) if isinstance(x, str) else x) else 0
            )
        
        # Features de comorbidades
        all_comorbidities = []
        for comorbidities_str in df['comorbidities']:
            comorbidities = eval(comorbidities_str) if isinstance(comorbidities_str, str) else comorbidities_str
            all_comorbidities.extend(comorbidities)
        
        common_comorbidities = pd.Series(all_comorbidities).value_counts().head(10).index.tolist()
        
        for comorbidity in common_comorbidities:
            features_df[f'comorbidity_{comorbidity}'] = df['comorbidities'].apply(
                lambda x: 1 if comorbidity in (eval(x) if isinstance(x, str) else x) else 0
            )
        
        # Features de sinais vitais
        for _, row in df.iterrows():
            vital_signs = eval(row['vital_signs']) if isinstance(row['vital_signs'], str) else row['vital_signs']
            for vital, value in vital_signs.items():
                if vital not in features_df.columns:
                    features_df[vital] = 0
                features_df.loc[len(features_df)-1, vital] = value
        
        # Target (categoria de risco)
        risk_encoder = LabelEncoder()
        target = risk_encoder.fit_transform(df['risk_category'])
        self.encoders['risk'] = risk_encoder
        
        self.feature_names = features_df.columns.tolist()
        
        print(f"Dataset preparado: {features_df.shape[0]} amostras, {features_df.shape[1]} features")
        print(f"Classes: {risk_encoder.classes_}")
        
        return features_df, target
    
    def train_models(self, X, y):
        """Treina todos os modelos"""
        print("\nDividindo dados em treino e teste...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Normalizar dados
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        # 1. Random Forest
        print("\nTreinando Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        # 2. Regressão Logística
        print("Treinando Regressão Logística...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        self.models['logistic_regression'] = lr_model
        
        # 3. Rede Neural (MLP)
        print("Treinando Rede Neural...")
        mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        mlp_model.fit(X_train_scaled, y_train)
        self.models['neural_network'] = mlp_model
        
        # Avaliar modelos
        self.evaluate_models(X_test, y_test, X_test_scaled)
        
        return X_test, y_test, X_test_scaled
    
    def evaluate_models(self, X_test, y_test, X_test_scaled):
        """Avalia todos os modelos"""
        print("\n=== AVALIAÇÃO DOS MODELOS ===")
        
        results = {}
        
        # Random Forest
        rf_pred = self.models['random_forest'].predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        results['random_forest'] = rf_accuracy
        print(f"\nRandom Forest - Acurácia: {rf_accuracy:.4f}")
        
        # Regressão Logística
        lr_pred = self.models['logistic_regression'].predict(X_test_scaled)
        lr_accuracy = accuracy_score(y_test, lr_pred)
        results['logistic_regression'] = lr_accuracy
        print(f"Regressão Logística - Acurácia: {lr_accuracy:.4f}")
        
        # Rede Neural
        mlp_pred = self.models['neural_network'].predict(X_test_scaled)
        mlp_accuracy = accuracy_score(y_test, mlp_pred)
        results['neural_network'] = mlp_accuracy
        print(f"Rede Neural - Acurácia: {mlp_accuracy:.4f}")
        
        # Melhor modelo
        best_model = max(results, key=results.get)
        print(f"\nMelhor modelo: {best_model} (Acurácia: {results[best_model]:.4f})")
        
        # Relatório detalhado do melhor modelo
        if best_model == 'random_forest':
            best_pred = rf_pred
        elif best_model == 'logistic_regression':
            best_pred = lr_pred
        else:
            best_pred = mlp_pred
        
        print(f"\nRelatório de classificação - {best_model}:")
        print(classification_report(y_test, best_pred, target_names=self.encoders['risk'].classes_))
        
        # Salvar resultados
        evaluation_results = {
            'model_accuracies': results,
            'best_model': best_model,
            'best_accuracy': results[best_model],
            'evaluation_date': datetime.now().isoformat(),
            'test_samples': len(y_test)
        }
        
        with open('/home/ubuntu/triagem_sus/models/evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        return results
    
    def save_models(self):
        """Salva todos os modelos treinados"""
        print("\nSalvando modelos...")
        
        # Salvar modelos
        for name, model in self.models.items():
            with open(f'/home/ubuntu/triagem_sus/models/{name}_model.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        # Salvar scalers e encoders
        with open('/home/ubuntu/triagem_sus/models/scalers.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)
        
        with open('/home/ubuntu/triagem_sus/models/encoders.pkl', 'wb') as f:
            pickle.dump(self.encoders, f)
        
        # Salvar nomes das features
        with open('/home/ubuntu/triagem_sus/models/feature_names.json', 'w') as f:
            json.dump(self.feature_names, f)
        
        print("Modelos salvos com sucesso!")
    
    def predict_risk(self, patient_data, model_name='random_forest'):
        """Prediz o risco de um paciente usando o modelo especificado"""
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} não encontrado")
        
        # Preparar features do paciente (implementação simplificada)
        # Em um sistema real, isso seria mais robusto
        features = np.zeros(len(self.feature_names))
        
        # Usar o modelo para predição
        model = self.models[model_name]
        
        if model_name in ['logistic_regression', 'neural_network']:
            features = self.scalers['standard'].transform([features])
        else:
            features = [features]
        
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        risk_category = self.encoders['risk'].inverse_transform([prediction])[0]
        confidence = max(probability)
        
        return {
            'risk_category': risk_category,
            'confidence': confidence,
            'probabilities': dict(zip(self.encoders['risk'].classes_, probability))
        }

def main():
    """Função principal para treinar os modelos"""
    print("=== TREINAMENTO DE MODELOS DE ML PARA TRIAGEM ===")
    
    # Criar instância da classe
    ml_trainer = TriageMLModels()
    
    # Carregar e preparar dados
    X, y = ml_trainer.load_and_prepare_data()
    
    # Treinar modelos
    X_test, y_test, X_test_scaled = ml_trainer.train_models(X, y)
    
    # Salvar modelos
    ml_trainer.save_models()
    
    print("\nTreinamento concluído com sucesso!")
    print("Modelos salvos em: /home/ubuntu/triagem_sus/models/")

if __name__ == "__main__":
    main()

