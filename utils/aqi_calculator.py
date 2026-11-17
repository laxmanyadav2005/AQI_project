"""
cpcb_aqi.py - Official CPCB (Central Pollution Control Board) Air Quality Index Calculator for India.
Implements the piecewise linear interpolation logic and breakpoint criteria.
"""

import math

# CPCB AQI categories, ranges, and color codes
AQI_CATEGORIES = [
    {"min": 0, "max": 50, "label": "Good", "color": "#00E400", "description": "Minimal impact"},
    {"min": 51, "max": 100, "label": "Satisfactory", "color": "#98F51F", "description": "Minor breathing discomfort to sensitive people"},
    {"min": 101, "max": 200, "label": "Moderate", "color": "#FF7E00", "description": "Breathing discomfort to the people with lungs, asthma and heart diseases"},
    {"min": 201, "max": 300, "label": "Poor", "color": "#FF0000", "description": "Breathing discomfort to most people on prolonged exposure"},
    {"min": 301, "max": 400, "label": "Very Poor", "color": "#8F3F97", "description": "Respiratory illness on prolonged exposure"},
    {"min": 401, "max": 5000, "label": "Severe", "color": "#7E0023", "description": "Affects healthy people and seriously impacts those with existing diseases"}
]

# Breakpoints for the 6 main pollutants
# Format: List of tuples (Concentration_Min, Concentration_Max, AQI_Min, AQI_Max)
BREAKPOINTS = {
    "PM2.5": [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (250, 1000, 401, 500)  # CPCB standard caps at 500 but linear extends for severe
    ],
    "PM10": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (430, 1000, 401, 500)
    ],
    "NO2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (400, 1000, 401, 500)
    ],
    "SO2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1600, 3000, 401, 500)
    ],
    "CO": [
        (0.0, 1.0, 0, 50),
        (1.1, 2.0, 51, 100),
        (2.1, 10.0, 101, 200),
        (10.1, 17.0, 201, 300),
        (17.1, 34.0, 301, 400),
        (34.0, 100.0, 401, 500)
    ],
    "O3": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (748, 1000, 401, 500)
    ]
}

def get_sub_index(pollutant, concentration):
    """
    Calculates sub-index (Ip) for a specific pollutant concentration using linear interpolation.
    Formula: Ip = ((I_hi - I_lo) / (B_hi - B_lo)) * (Cp - B_lo) + I_lo
    """
    if concentration is None or math.isnan(concentration) or concentration < 0:
        return None
        
    if pollutant not in BREAKPOINTS:
        raise ValueError(f"Unknown pollutant: {pollutant}. Supported: {list(BREAKPOINTS.keys())}")
        
    for b_lo, b_hi, i_lo, i_hi in BREAKPOINTS[pollutant]:
        if b_lo <= concentration <= b_hi:
            # Linear interpolation formula
            ip = ((i_hi - i_lo) / (b_hi - b_lo)) * (concentration - b_lo) + i_lo
            return round(ip)
            
    # Handle values above maximum defined breakpoints
    last_breakpoint = BREAKPOINTS[pollutant][-1]
    b_lo, b_hi, i_lo, i_hi = last_breakpoint
    if concentration > b_hi:
        # Extrapolate using the final slope
        ip = ((i_hi - i_lo) / (b_hi - b_lo)) * (concentration - b_lo) + i_lo
        return min(round(ip), 500) # Cap at 500 as per official standards
        
    return None

def get_aqi_category(aqi_value):
    """
    Returns the category, color hex, and health description for a given AQI value.
    """
    if aqi_value is None or math.isnan(aqi_value):
        return None
        
    for cat in AQI_CATEGORIES:
        if cat["min"] <= aqi_value <= cat["max"]:
            return cat
            
    return AQI_CATEGORIES[-1] # Default to Severe if above 500

def calculate_aqi(pollutant_data):
    """
    Calculates the final overall Air Quality Index (AQI) based on a dictionary of pollutant concentrations.
    Rules:
    1. Must have at least 3 pollutants present.
    2. At least one of PM2.5 or PM10 must be present.
    3. Final AQI is the maximum of all individual sub-indices.
    
    pollutant_data format: {'PM2.5': 45.0, 'PM10': 110.0, 'SO2': 12.0, ...}
    """
    sub_indices = {}
    valid_pollutants = 0
    has_pm = False
    
    for pollutant, conc in pollutant_data.items():
        if pollutant in BREAKPOINTS and conc is not None:
            sub_idx = get_sub_index(pollutant, conc)
            if sub_idx is not None:
                sub_indices[pollutant] = sub_idx
                valid_pollutants += 1
                if pollutant in ["PM2.5", "PM10"]:
                    has_pm = True
                    
    # Validation checks
    if valid_pollutants < 3:
        return {
            "status": "error",
            "message": f"Calculation requires at least 3 pollutants (Got {valid_pollutants})."
        }
        
    if not has_pm:
        return {
            "status": "error",
            "message": "Calculation requires at least one particulate matter (PM2.5 or PM10)."
        }
        
    final_aqi = max(sub_indices.values())
    category_info = get_aqi_category(final_aqi)
    
    return {
        "status": "success",
        "aqi": final_aqi,
        "category": category_info["label"],
        "color": category_info["color"],
        "description": category_info["description"],
        "sub_indices": sub_indices
    }

# Self-test block
if __name__ == "__main__":
    print("Testing CPCB AQI Calculator...")
    
    test_cases = [
        # Test Case 1: Standard satisfactory day
        {"PM2.5": 25.0, "PM10": 75.0, "NO2": 15.0},
        # Test Case 2: Moderate day with PM10 high
        {"PM2.5": 45.0, "PM10": 120.0, "SO2": 15.0, "CO": 1.5},
        # Test Case 3: Poor day (severe PM2.5)
        {"PM2.5": 110.0, "PM10": 210.0, "O3": 45.0},
        # Test Case 4: Insufficient pollutants
        {"PM2.5": 55.0, "PM10": 110.0}
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nCase {i}: Inputs = {case}")
        res = calculate_aqi(case)
        if res["status"] == "success":
            print(f"  AQI: {res['aqi']} ({res['category']})")
            print(f"  Sub-indices: {res['sub_indices']}")
        else:
            print(f"  Error: {res['message']}")
