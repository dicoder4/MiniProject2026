"""
Real-time Risk Assessment Module
Enhanced with Colab-level sophistication
"""

import numpy as np
import pandas as pd

def calculate_risk_level(flooded_people_count, total_people, evacuation_success_rate=None):
    """Calculate risk level with enhanced logic from Colab"""
    
    if total_people == 0:
        return "🟢 LOW RISK", 0
    
    # Population at risk percentage
    population_at_risk_pct = flooded_people_count / total_people * 100
    
    # Base risk assessment
    if population_at_risk_pct > 50:
        base_risk = "🔴 HIGH RISK"
        risk_score = 3
    elif population_at_risk_pct > 20:
        base_risk = "🟡 MEDIUM RISK"
        risk_score = 2
    else:
        base_risk = "🟢 LOW RISK"
        risk_score = 1
    
    # Adjust based on evacuation success rate if available
    if evacuation_success_rate is not None:
        if evacuation_success_rate < 70:
            risk_score = min(3, risk_score + 1)
        elif evacuation_success_rate > 90:
            risk_score = max(1, risk_score - 1)
    
    # Final risk determination
    if risk_score >= 3:
        return "🔴 HIGH RISK", population_at_risk_pct
    elif risk_score == 2:
        return "🟡 MEDIUM RISK", population_at_risk_pct
    else:
        return "🟢 LOW RISK", population_at_risk_pct

def generate_risk_recommendations(risk_level, population_at_risk_pct, evacuation_success_rate=None):
    """Generate actionable recommendations based on risk assessment"""
    
    recommendations = []
    
    if "HIGH RISK" in risk_level:
        recommendations.extend([
            "🚨 IMMEDIATE ACTION REQUIRED",
            "• Activate emergency evacuation protocols",
            "• Deploy additional rescue teams",
            "• Set up emergency shelters",
            "• Issue public emergency alerts",
            "• Coordinate with emergency services"
        ])
    elif "MEDIUM RISK" in risk_level:
        recommendations.extend([
            "⚠️ ENHANCED MONITORING REQUIRED",
            "• Prepare evacuation resources",
            "• Monitor flood progression closely",
            "• Alert emergency services",
            "• Prepare public notifications",
            "• Review evacuation routes"
        ])
    else:
        recommendations.extend([
            "✅ SITUATION MANAGEABLE",
            "• Continue monitoring",
            "• Maintain readiness",
            "• Regular status updates",
            "• Keep evacuation plans current"
        ])
    
    if evacuation_success_rate is not None and evacuation_success_rate < 80:
        recommendations.append("• Review and optimize evacuation routes")
        recommendations.append("• Consider alternative safe centers")
    
    return recommendations
