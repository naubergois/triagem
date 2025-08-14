# tests/test_patient_analysis.py
import os, csv, json, re, pytest
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.app import simulate_triage

try:
    from agents.patient_analysis_agent import PatientAnalysisAgent
except Exception:
    PatientAnalysisAgent = None

def parse_score(text: str) -> str:
    m = re.search(r"Pontuação:\s*(\d+(?:\.\d+)?)", text)
    return m.group(1) if m else "N/A"

def build_patient_record(row: dict) -> dict:
    systolic, diastolic = [int(x) for x in row["pressao_arterial"].split("/")]
    symptoms = [s.strip() for s in row["sintomas"].split(",") if s.strip()]
    return {
        "patient_id": row["nome"],
        "symptoms": symptoms,
        "comorbidities": [s.strip() for s in row["historico_medico"].split(",") if s.strip()],
        "vital_signs": {
            "temperature": float(row["temperatura"]),
            "heart_rate": int(row["frequencia_cardiaca"]),
            "blood_pressure_systolic": systolic,
            "blood_pressure_diastolic": diastolic,
            "respiratory_rate": 16,
            "oxygen_saturation": int(row["saturacao_oxigenio"]),
        },
    }

def load_all_rows():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "examples", "example_patients.csv")
    with open(csv_path, newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)

@pytest.mark.integration
@pytest.mark.parametrize("row", list(load_all_rows()))
def test_patient_analysis_each_case(row, tmp_path):
    load_dotenv()
    if PatientAnalysisAgent is None:
        pytest.skip("Dependências do PatientAnalysisAgent indisponíveis")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY não definido no .env")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "evaluation")
    os.makedirs(out_dir, exist_ok=True)

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

    # salva um arquivo por paciente
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", metrics["patient"])
    with open(os.path.join(out_dir, f"{safe_name}.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    assert score != "N/A"
