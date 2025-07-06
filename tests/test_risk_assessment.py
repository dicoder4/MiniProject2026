import pytest
from risk_assessment import calculate_risk_level, generate_risk_recommendations

def test_calculate_risk_level():
    # Test high risk scenario
    level, score = calculate_risk_level(80, 100)
    assert "HIGH RISK" in level
    
    # Test medium risk scenario
    level, score = calculate_risk_level(30, 100)
    assert "MEDIUM RISK" in level
    
    # Test low risk scenario
    level, score = calculate_risk_level(5, 100)
    assert "LOW RISK" in level
    
    # Test edge case
    level, score = calculate_risk_level(0, 0)
    assert level is not None

def test_risk_recommendations():
    # Test high risk recommendations
    high_risk_rec = generate_risk_recommendations("HIGH RISK", 75)
    assert any("emergency evacuation" in rec.lower() for rec in high_risk_rec)
    
    # Test medium risk recommendations
    med_risk_rec = generate_risk_recommendations("MEDIUM RISK", 30)
    assert any("prepare" in rec.lower() for rec in med_risk_rec)
    
    # Test low risk recommendations
    low_risk_rec = generate_risk_recommendations("LOW RISK", 5)
    assert any("monitor" in rec.lower() for rec in low_risk_rec)