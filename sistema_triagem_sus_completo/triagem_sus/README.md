# Sistema de Triagem Inteligente SUS

## Descrição

Sistema baseado em agentes inteligentes para triagem de pacientes no Sistema Único de Saúde (SUS), utilizando LangChain e modelos de Machine Learning para classificação automática de risco conforme o Protocolo de Manchester.

## Características Principais

- **Agentes Inteligentes**: Implementados com LangChain para análise automática de dados clínicos
- **Machine Learning**: Modelos treinados (Random Forest, Regressão Logística, Redes Neurais) com acurácia >93%
- **Protocolo de Manchester**: Classificação em 5 categorias de risco (Vermelho, Laranja, Amarelo, Verde, Azul)
- **Interface Web**: Sistema completo com formulários de triagem e dashboard
- **API REST**: Endpoints para integração com outros sistemas

## Estrutura do Projeto

```
triagem_sus/
├── data/                          # Dados e datasets
│   ├── generate_synthetic_data.py # Gerador de dados sintéticos
│   ├── patients_data.csv         # Dataset de pacientes
│   ├── patients_data.json        # Dataset em formato JSON
│   ├── metadata.json             # Metadados do dataset
│   └── triage_results.json       # Resultados de triagem
├── models/                        # Modelos de Machine Learning
│   ├── ml_models.py              # Implementação dos modelos ML
│   ├── evaluation_results.json   # Resultados da avaliação
│   ├── random_forest_model.pkl   # Modelo Random Forest treinado
│   ├── logistic_regression_model.pkl # Modelo Regressão Logística
│   ├── neural_network_model.pkl  # Modelo Rede Neural
│   ├── scalers.pkl               # Normalizadores
│   ├── encoders.pkl              # Codificadores
│   └── feature_names.json        # Nomes das features
├── agents/                        # Agentes Inteligentes
│   └── intelligent_triage_agent.py # Agente principal de triagem
├── web/                          # Interface Web
│   ├── app.py                    # Aplicação Flask
│   └── templates/                # Templates HTML
│       ├── base.html
│       ├── index.html
│       ├── triagem.html
│       ├── resultado.html
│       └── dashboard.html
├── docs/                         # Documentação
├── notebooks/                    # Jupyter Notebooks (análises)
├── requirements.txt              # Dependências Python
└── README.md                     # Este arquivo
```

## Instalação e Configuração

### Pré-requisitos

- Python 3.11+
- pip

### Instalação

1. Clone ou extraia o projeto:
```bash
cd triagem_sus
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Gere os dados sintéticos (se necessário):
```bash
python data/generate_synthetic_data.py
```

4. Treine os modelos de ML:
```bash
python models/ml_models.py
```

5. (Opcional) Para treinar o detector de pneumonia, baixe o dataset
   ["Chest X-Ray Images (Pneumonia)"](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
   do Kaggle e extraia as pastas `train`, `val` e `test` em `data/chest_xray/`.
   A estrutura final deve ficar assim:

   ```
   data/chest_xray/
   ├── train/
   ├── val/
   └── test/
   ```

## Uso

### 1. Agente Inteligente de Triagem

Execute o agente para triagem automática:

```bash
python agents/intelligent_triage_agent.py
```
Se houver um arquivo `.env` com `OPENAI_API_KEY`, o agente também exibirá uma
avaliação clínica gerada via OpenAI juntamente com a classificação de risco.

### 2. Interface Web

Inicie a aplicação web:

```bash
cd web
python app.py
```

Acesse: `http://localhost:5000`

### 3. API REST

Endpoint para triagem via API:

```bash
curl -X POST http://localhost:5000/api/triagem \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "PAC_001",
    "age_range": "45-60",
    "gender": "M",
    "symptoms": ["dor torácica", "dificuldade respiratória"],
    "comorbidities": ["diabetes"],
    "vital_signs": {
      "temperature": 37.5,
      "heart_rate": 95,
      "blood_pressure_systolic": 140,
      "blood_pressure_diastolic": 90,
      "respiratory_rate": 20,
      "oxygen_saturation": 96
    },
  "chief_complaint": "Dor no peito há 2 horas"
}'
```

### 4. Agente de Análise Médica

Para obter uma avaliação resumida de um paciente com pontuação de gravidade
utilizando a API da OpenAI, crie um arquivo `.env` com a variável
`OPENAI_API_KEY` e execute:

```bash
python agents/patient_analysis_agent.py
```

### 5. Exemplo Rápido

Para uma demonstração simples, execute o script em `examples/quickstart.py`.
Ele utiliza um pequeno conjunto de pacientes de amostra e permite incluir uma
nova triagem interativamente. Assim que os dados do novo paciente são
informados, o agente gera uma avaliação imediata. Todos os resultados são
avaliados com a OpenAI e um gráfico de pontuação é gerado. Certifique-se de ter
a variável de ambiente `OPENAI_API_KEY` definida:

```bash
python examples/quickstart.py
```

## Protocolo de Manchester

O sistema implementa o protocolo internacional de triagem com as seguintes categorias:

| Categoria | Cor      | Prioridade | Tempo Máximo | Descrição     |
|-----------|----------|------------|--------------|---------------|
| VERMELHO  | Vermelho | 1          | 0 min        | Emergência    |
| LARANJA   | Laranja  | 2          | 10 min       | Muito urgente |
| AMARELO   | Amarelo  | 3          | 60 min       | Urgente       |
| VERDE     | Verde    | 4          | 120 min      | Pouco urgente |
| AZUL      | Azul     | 5          | 240 min      | Não urgente   |

## Modelos de Machine Learning

### Algoritmos Implementados

1. **Random Forest**: Melhor performance (93.1% acurácia)
2. **Regressão Logística**: 93.1% acurácia
3. **Rede Neural (MLP)**: 92.4% acurácia

### Features Utilizadas

- **Demográficas**: Idade, sexo
- **Sintomas**: One-hot encoding dos 20 sintomas mais comuns
- **Comorbidades**: One-hot encoding das 10 comorbidades mais comuns
- **Sinais Vitais**: Temperatura, FC, PA, FR, SatO2

### Avaliação

Os modelos foram avaliados usando:
- Divisão treino/teste (80/20)
- Validação cruzada
- Métricas: Acurácia, Precisão, Recall, F1-Score
- Matriz de confusão

## Agentes Inteligentes

### Arquitetura

O sistema utiliza LangChain para implementar agentes inteligentes com:

- **Ferramentas especializadas**:
  - `get_patient_data`: Busca dados do paciente
  - `classify_manchester_risk`: Classifica risco pelo protocolo
  - `generate_recommendations`: Gera recomendações clínicas
  - `check_alerts`: Verifica alertas especiais

- **Padrão ReAct**: Reasoning and Acting para tomada de decisão

- **Memória**: Contexto persistente durante a análise

### Funcionalidades

- Análise automática de sintomas e sinais vitais
- Classificação de risco baseada em regras e ML
- Geração de recomendações personalizadas
- Alertas para casos especiais (idosos, comorbidades)
- Justificativas detalhadas das decisões

## Interface Web

### Páginas Disponíveis

1. **Página Inicial** (`/`): Visão geral do sistema
2. **Nova Triagem** (`/triagem`): Formulário de entrada de dados
3. **Resultado** (`/resultado`): Exibição do resultado da triagem
4. **Dashboard** (`/dashboard`): Estatísticas e histórico

### Características

- Design responsivo (Bootstrap 5)
- Formulários intuitivos para entrada de dados
- Visualização clara dos resultados
- Códigos de cores conforme protocolo de Manchester
- Funcionalidade de impressão

## Dataset Sintético

### Características

- **10.000 pacientes** sintéticos
- **Distribuição realística** por categoria de risco
- **Sintomas baseados** em literatura médica
- **Sinais vitais** correlacionados com gravidade
- **Comorbidades** por faixa etária

### Geração

O dataset é gerado automaticamente com:
- Protocolo de Manchester como base
- Distribuição estatística realística
- Correlações clínicas apropriadas
- Variabilidade controlada

## Contribuição

### Estrutura de Desenvolvimento

1. **Dados**: Geração e preparação em `data/`
2. **Modelos**: Implementação ML em `models/`
3. **Agentes**: Lógica de negócio em `agents/`
4. **Web**: Interface em `web/`
5. **Testes**: Validação em `tests/` (a implementar)

### Próximos Passos

- [ ] Integração com APIs do SUS (SISREG, e-SUS)
- [ ] Implementação de testes automatizados
- [ ] Otimização de performance
- [ ] Validação com dados reais
- [ ] Deploy em produção
- [ ] Monitoramento e logging
- [ ] Auditoria e compliance

## Licença

Este projeto foi desenvolvido para fins acadêmicos e de pesquisa, baseado no artigo "Utilização de agentes inteligentes na triagem de pacientes no sistema público de saúde".

## Autores

- **Implementação**: Sistema desenvolvido com base no artigo de Marx Haron Gomes Barbosa e Francisco Nauber Bernardo Gois
- **Tecnologias**: LangChain, scikit-learn, Flask, Bootstrap

## Contato

Para dúvidas ou sugestões sobre a implementação, consulte a documentação técnica ou os comentários no código.

---

**Nota**: Este é um sistema de demonstração acadêmica. Para uso em produção, são necessárias validações adicionais, certificações médicas e compliance com regulamentações de saúde.

