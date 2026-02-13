"""
F1 2026 Season Predictor — Configuration
"""

# 2026 Driver-Team lineup
LINEUP_2026 = {
    "Alpine": ["Pierre Gasly", "Franco Colapinto"],
    "Aston Martin": ["Fernando Alonso", "Lance Stroll"],
    "Audi": ["Niko Hulkenberg", "Gabriel Bortoleto"],
    "Cadillac": ["Sergio Perez", "Valtteri Bottas"],
    "Ferrari": ["Charles Leclerc", "Lewis Hamilton"],
    "Haas F1 Team": ["Esteban Ocon", "Oliver Bearman"],
    "McLaren": ["Lando Norris", "Oscar Piastri"],
    "Mercedes": ["George Russell", "Kimi Antonelli"],
    "Racing Bulls": ["Liam Lawson", "Arvid Lindblad"],
    "Red Bull Racing": ["Max Verstappen", "Isack Hadjar"],
    "Williams": ["Carlos Sainz", "Alexander Albon"],
}

# Map driver names to FastF1 abbreviations / identifiers used historically
# This helps match drivers across seasons where names may differ
DRIVER_NAME_MAP = {
    "Pierre Gasly": ["GAS", "Pierre Gasly"],
    "Franco Colapinto": ["COL", "Franco Colapinto"],
    "Fernando Alonso": ["ALO", "Fernando Alonso"],
    "Lance Stroll": ["STR", "Lance Stroll"],
    "Niko Hulkenberg": ["HUL", "Nico Hulkenberg", "Nico Hülkenberg", "Niko Hulkenberg"],
    "Gabriel Bortoleto": ["BOR", "Gabriel Bortoleto"],
    "Sergio Perez": ["PER", "Sergio Perez", "Sergio Pérez"],
    "Valtteri Bottas": ["BOT", "Valtteri Bottas"],
    "Charles Leclerc": ["LEC", "Charles Leclerc"],
    "Lewis Hamilton": ["HAM", "Lewis Hamilton"],
    "Esteban Ocon": ["OCO", "Esteban Ocon"],
    "Oliver Bearman": ["BEA", "Oliver Bearman"],
    "Lando Norris": ["NOR", "Lando Norris"],
    "Oscar Piastri": ["PIA", "Oscar Piastri"],
    "George Russell": ["RUS", "George Russell"],
    "Kimi Antonelli": ["ANT", "Kimi Antonelli", "Andrea Kimi Antonelli"],
    "Liam Lawson": ["LAW", "Liam Lawson"],
    "Arvid Lindblad": ["LIN", "Arvid Lindblad"],
    "Max Verstappen": ["VER", "Max Verstappen"],
    "Isack Hadjar": ["HAD", "Isack Hadjar"],
    "Carlos Sainz": ["SAI", "Carlos Sainz"],
    "Alexander Albon": ["ALB", "Alexander Albon", "Alex Albon"],
}

# Drivers with no/minimal F1 history — we'll use proxy ratings
ROOKIES_2026 = ["Arvid Lindblad"]

# Historical seasons to train on (complete seasons only)
TRAINING_SEASONS = list(range(2019, 2026))  # 2019-2025

# Seasons with major regulation changes (for adaptability analysis)
REGULATION_CHANGE_YEARS = {
    2022: "Ground effect aero",
    2026: "New PU + aero regs",
}

# Team name mapping across seasons (teams rename/rebrand)
TEAM_NAME_MAP = {
    "Alpine": ["Alpine", "Alpine F1 Team", "Renault"],
    "Aston Martin": ["Aston Martin", "Racing Point", "Force India"],
    "Audi": ["Sauber", "Alfa Romeo", "Alfa Romeo Racing", "Kick Sauber"],
    "Cadillac": [],  # New team — no history
    "Ferrari": ["Ferrari", "Scuderia Ferrari"],
    "Haas F1 Team": ["Haas F1 Team", "Haas", "MoneyGram Haas F1 Team"],
    "McLaren": ["McLaren", "McLaren F1 Team"],
    "Mercedes": ["Mercedes", "Mercedes-AMG Petronas F1 Team"],
    "Racing Bulls": ["RB", "AlphaTauri", "Toro Rosso", "Racing Bulls"],
    "Red Bull Racing": ["Red Bull Racing", "Red Bull", "Oracle Red Bull Racing"],
    "Williams": ["Williams", "Williams Racing"],
}

# Points system
F1_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

# 2026 calendar (expected — using 2025 as proxy with likely adjustments)
CALENDAR_2026 = [
    "Australia", "China", "Japan", "Bahrain", "Saudi Arabia",
    "Miami", "Emilia Romagna", "Monaco", "Spain", "Canada",
    "Austria", "Great Britain", "Belgium", "Hungary", "Netherlands",
    "Italy", "Azerbaijan", "Singapore", "United States", "Mexico",
    "Brazil", "Las Vegas", "Qatar", "Abu Dhabi",
]

# Circuit type classification
# 0=street, 1=high_speed, 2=technical, 3=balanced
CIRCUIT_TYPE_NAMES = {0: "street", 1: "high_speed", 2: "technical", 3: "balanced"}

CALENDAR_2026_TYPES = {
    "Australia": 3, "China": 2, "Japan": 2, "Bahrain": 3, "Saudi Arabia": 0,
    "Miami": 0, "Emilia Romagna": 3, "Monaco": 0, "Spain": 2, "Canada": 1,
    "Austria": 3, "Great Britain": 1, "Belgium": 1, "Hungary": 2, "Netherlands": 2,
    "Italy": 1, "Azerbaijan": 0, "Singapore": 0, "United States": 3, "Mexico": 3,
    "Brazil": 3, "Las Vegas": 0, "Qatar": 3, "Abu Dhabi": 3,
}

# Keywords for classifying historical event names
_STREET_KW = ["monaco", "singapore", "azerbaijan", "baku", "las vegas", "miami", "jeddah", "saudi"]
_HIGHSPEED_KW = ["monza", "italian", "belgium", "belgian", "spa", "great britain", "british",
                 "silverstone", "canada", "canadian", "montreal"]
_TECHNICAL_KW = ["hungary", "hungarian", "spain", "spanish", "barcelona", "netherlands", "dutch",
                 "zandvoort", "japan", "japanese", "suzuka", "china", "chinese", "shanghai"]


def classify_circuit_type(event_name):
    """Classify a race event into circuit type (0=street, 1=high_speed, 2=technical, 3=balanced)."""
    e = event_name.lower()
    for kw in _STREET_KW:
        if kw in e:
            return 0
    for kw in _HIGHSPEED_KW:
        if kw in e:
            return 1
    for kw in _TECHNICAL_KW:
        if kw in e:
            return 2
    return 3
