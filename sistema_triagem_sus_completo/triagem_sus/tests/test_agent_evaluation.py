import os
import sys
import csv
import json
import re
import pytest
from dotenv import load_dotenv

# Ensure project root in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.app import simulate_triage  # noqa: E402

# Try to import patient analysis agent, skip if dependencies missing
try:  # noqa: E402
    from agents.patient_analysis_agent import PatientAnalysisAgent
except Exception:  # pragma: no cover
    PatientAnalysisAgent = None


def parse_score(text: str) -> str:
    """Extrai a pontuação numérica do texto de análise."""
    match = re.search(r"Pontuação:\s*(\d+(?:\.\d+)?)", text)
    return match.group(1) if match else "N/A"


def build_patient_record(row: dict) -> dict:
    """Converte uma linha CSV em registro de paciente."""
    systolic, diastolic = [int(x) for x in row["pressao_arterial"].split("/")]
    symptoms = [s.strip() for s in row["sintomas"].split(",")]
    return {
        "patient_id": row["nome"],
        "symptoms": symptoms,
        "comorbidities": [s.strip() for s in row["historico_medico"].split(",")],
        "vital_signs": {
            "temperature": float(row["temperatura"]),
            "heart_rate": int(row["frequencia_cardiaca"]),
            "blood_pressure_systolic": systolic,
            "blood_pressure_diastolic": diastolic,
            "respiratory_rate": 16,
            "oxygen_saturation": int(row["saturacao_oxigenio"]),
        },
    }


@pytest.mark.integration
def test_patient_analysis_metrics(tmp_path):
    """Executa o agente de análise e salva métricas em arquivo."""
    load_dotenv()
    if PatientAnalysisAgent is None:
        pytest.skip("Dependências do PatientAnalysisAgent indisponíveis")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY não definido no .env")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "examples", "example_patients.csv")
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    record = build_patient_record(row)

    triage_result = simulate_triage(record)
    agent = PatientAnalysisAgent()
    analysis = agent.evaluate(record)
    score = parse_score(analysis)

    metrics = {
        "patient": row["nome"],
        "risk_category": triage_result["risk_category"],
        "priority": triage_result["priority"],
        "score": score,
        "analysis": analysis,
    }

    out_path = os.path.join(base_dir, "evaluation", "agent_metrics.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    assert score != "N/A"
