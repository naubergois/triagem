import os
import sys
import pytest

# Ensure the project root is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.app import simulate_triage


@pytest.mark.parametrize(
    "symptoms,vitals,expected_category,expected_priority",
    [
        (
            ["dor torácica"],
            {
                "temperature": 40.0,
                "heart_rate": 80,
                "blood_pressure_systolic": 120,
                "blood_pressure_diastolic": 80,
                "respiratory_rate": 16,
                "oxygen_saturation": 90,
            },
            "VERMELHO",
            1,
        ),
        (
            ["febre alta"],
            {
                "temperature": 37.0,
                "heart_rate": 130,
                "blood_pressure_systolic": 120,
                "blood_pressure_diastolic": 80,
                "respiratory_rate": 16,
                "oxygen_saturation": 96,
            },
            "AMARELO",
            3,
        ),
        (
            ["dor de cabeça"],
            {
                "temperature": 36.5,
                "heart_rate": 70,
                "blood_pressure_systolic": 120,
                "blood_pressure_diastolic": 80,
                "respiratory_rate": 16,
                "oxygen_saturation": 99,
            },
            "AZUL",
            5,
        ),
    ],
)
def test_simulate_triage(symptoms, vitals, expected_category, expected_priority):
    """Ensure that simulate_triage categorizes risk appropriately."""
    patient_data = {"symptoms": symptoms, "vital_signs": vitals}
    result = simulate_triage(patient_data)
    assert result["risk_category"] == expected_category
    assert result["priority"] == expected_priority
