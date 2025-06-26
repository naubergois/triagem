#!/bin/bash

echo "=== Sistema de Triagem Inteligente SUS ==="
echo "Script de Instalação e Configuração"
echo ""

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "ERRO: Python 3 não encontrado. Instale Python 3.11+ antes de continuar."
    exit 1
fi

echo "✓ Python encontrado: $(python3 --version)"

# Criar ambiente virtual (opcional)
read -p "Deseja criar um ambiente virtual? (y/n): " create_venv
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo "Criando ambiente virtual..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Ambiente virtual criado e ativado"
fi

# Instalar dependências
echo "Instalando dependências..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependências instaladas com sucesso"
else
    echo "ERRO: Falha na instalação das dependências"
    exit 1
fi

# Gerar dados sintéticos
echo "Gerando dados sintéticos..."
python3 data/generate_synthetic_data.py

if [ $? -eq 0 ]; then
    echo "✓ Dados sintéticos gerados"
else
    echo "ERRO: Falha na geração dos dados"
    exit 1
fi

# Treinar modelos
echo "Treinando modelos de Machine Learning..."
python3 models/ml_models.py

if [ $? -eq 0 ]; then
    echo "✓ Modelos treinados com sucesso"
else
    echo "ERRO: Falha no treinamento dos modelos"
    exit 1
fi

# Testar agente
echo "Testando agente inteligente..."
python3 agents/intelligent_triage_agent.py

if [ $? -eq 0 ]; then
    echo "✓ Agente testado com sucesso"
else
    echo "ERRO: Falha no teste do agente"
    exit 1
fi

echo ""
echo "=== INSTALAÇÃO CONCLUÍDA ==="
echo ""
echo "Para iniciar a aplicação web:"
echo "  cd web"
echo "  python3 app.py"
echo ""
echo "Acesse: http://localhost:5000"
echo ""
echo "Para testar a API:"
echo "  curl -X POST http://localhost:5000/api/triagem -H 'Content-Type: application/json' -d '{...}'"
echo ""
echo "Documentação completa: README.md"

