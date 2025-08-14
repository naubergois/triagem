"""Gera um relatório de avaliação de triagem usando um agente LLM."""
import csv
import os
import re
import sys
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.app import simulate_triage

try:
    from agents.patient_analysis_agent import PatientAnalysisAgent
except Exception as exc:  # pylint: disable=broad-except
    PatientAnalysisAgent = None  # type: ignore
    ANALYSIS_ERROR = str(exc)
else:
    ANALYSIS_ERROR = ""


def parse_score(text: str) -> str:
    """Extrai a pontuação numérica do texto de análise."""
    match = re.search(r"Pontuação:\s*(\d+(?:\.\d+)?)", text)
    return match.group(1) if match else "N/A"


def load_patients(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_patient_record(row: Dict[str, str]) -> Dict:
    """Converte uma linha do CSV para o formato esperado pela triagem."""
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


def main() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "examples", "example_patients.csv")
    patients = load_patients(csv_path)

    agent = None
    if PatientAnalysisAgent is not None:
        try:
            agent = PatientAnalysisAgent()
        except Exception as exc:  # pylint: disable=broad-except
            agent = None
            global ANALYSIS_ERROR
            ANALYSIS_ERROR = str(exc)

    results = []
    for row in patients:
        record = build_patient_record(row)
        triage_result = simulate_triage(record)
        analysis = ""
        score = "N/A"
        if agent is not None:
            try:
                analysis = agent.evaluate(record)
                score = parse_score(analysis)
            except Exception as exc:  # pylint: disable=broad-except
                analysis = f"Erro na análise: {exc}"
        else:
            analysis = f"Agente indisponível: {ANALYSIS_ERROR}"

        results.append({
            "Paciente": row["nome"],
            "Risco": triage_result["risk_category"],
            "Prioridade": triage_result["priority"],
            "Pontuação": score,
            "Avaliação": analysis.replace("|", "\\|")
        })

    report_lines = ["# Resultados da Triagem\n", "\n", "| Paciente | Categoria de Risco | Prioridade | Pontuação | Avaliação |\n", "|---|---|---|---|---|\n"]
    for r in results:
        report_lines.append(f"| {r['Paciente']} | {r['Risco']} | {r['Prioridade']} | {r['Pontuação']} | {r['Avaliação']} |\n")

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "triage_report.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    print(f"Relatório salvo em {output_path}")


if __name__ == "__main__":
    main()
