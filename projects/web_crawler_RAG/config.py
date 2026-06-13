"""
Central config: US states list, field definitions, and search query templates.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID  = os.getenv("GOOGLE_CSE_ID", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
SERPAPI_KEY    = os.getenv("SERPAPI_KEY", "")
DB_PATH        = os.getenv("DB_PATH", "./pbc_data.db")
MAX_RESULTS    = int(os.getenv("MAX_RESULTS", "5"))
SEARCH_DELAY   = float(os.getenv("SEARCH_DELAY", "1.0"))

US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming",
]

# State abbreviation map for queries
STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}

# 2024 US Census state population estimates (for per-capita calculations)
STATE_POPULATION = {
    "Alabama": 5108468, "Alaska": 733406, "Arizona": 7431344, "Arkansas": 3067732,
    "California": 39538223, "Colorado": 5773714, "Connecticut": 3605944,
    "Delaware": 1003384, "Florida": 21538187, "Georgia": 10711908,
    "Hawaii": 1455271, "Idaho": 1939033, "Illinois": 12812508, "Indiana": 6785528,
    "Iowa": 3190369, "Kansas": 2937880, "Kentucky": 4509394, "Louisiana": 4657757,
    "Maine": 1395722, "Maryland": 6177224, "Massachusetts": 7029917,
    "Michigan": 10077331, "Minnesota": 5706494, "Mississippi": 2961279,
    "Missouri": 6154913, "Montana": 1122867, "Nebraska": 1961504,
    "Nevada": 3104614, "New Hampshire": 1377529, "New Jersey": 9288994,
    "New Mexico": 2117522, "New York": 20201249, "North Carolina": 10439388,
    "North Dakota": 779094, "Ohio": 11799448, "Oklahoma": 3959353,
    "Oregon": 4237256, "Pennsylvania": 13002700, "Rhode Island": 1097379,
    "South Carolina": 5118425, "South Dakota": 886667, "Tennessee": 6910840,
    "Texas": 29145505, "Utah": 3271616, "Vermont": 643077,
    "Virginia": 8631393, "Washington": 7705281, "West Virginia": 1793716,
    "Wisconsin": 5893718, "Wyoming": 576851,
}

# ── Search query templates ────────────────────────────────────────────────────
# Each field has 2-3 query variants; the agent tries them in order until it
# finds extractable data.  {state} is substituted at runtime.
SEARCH_QUERIES = {
    "pbc_patients_adults": [
        'Primary Biliary Cholangitis prevalence {state} adults estimated patients 2023 2024',
        'PBC patients {state} epidemiology incidence prevalence adults',
        '"primary biliary cholangitis" {state} diagnosed patients statistics',
    ],
    "second_line_eligible": [
        'PBC UDCA inadequate response {state} second-line eligible patients estimate',
        '"primary biliary Cholangitis" 2L eligible {state} treatment failure UDCA',
        'obeticholic acid eligibility {state} PBC second line therapy patients',
    ],
    "hepatology_density": [
        'hepatologist density {state} per million population specialists 2022 2023',
        'gastroenterologist hepatologist workforce {state} count per capita',
        'ACGME fellowship hepatology physicians {state} practicing',
    ],
    "medicare_pct": [
        'Medicare enrollment {state} percentage adults 2023 CMS',
        'CMS Medicare beneficiaries {state} percentage insured population',
        'Medicare Part D {state} liver disease chronic condition enrollment',
    ],
    "dominant_payer": [
        'health insurance market share {state} 2023 commercial insurer largest',
        'Cigna Aetna BCBS UnitedHealth {state} enrollment market share largest insurer',
        '{state} dominant health plan enrollment commercial insurance 2023',
    ],
    "second_line_share_lean": [
        'PBC second line treatment market share {state} IQIRVO LIVDELZI Ocaliva 2024',
        'primary biliary cholangitis drug prescription share {state} 2024',
        'PBC market {state} treatment landscape obeticholic acid elafibranor seladelpar',
    ],
    "ex_ocaliva": [
        'Ocaliva obeticholic acid PBC {state} withdrawal discontinuation patients switch',
        'OCA Ocaliva FDA withdrawal PBC {state} prescription market 2024',
        'obeticholic acid discontinuation {state} switch therapy PBC 2024',
    ],
    "iqirvo_share": [
        'IQIRVO elafibranor prescription share {state} PBC 2024 Ipsen',
        'elafibranor IQIRVO market share launch {state} primary biliary cholangitis',
        'Ipsen IQIRVO {state} formulary adoption payer coverage',
    ],
    "livdelzi_share": [
        'LIVDELZI seladelpar prescription share {state} PBC 2024 CymaBay',
        'seladelpar LIVDELZI market share launch {state} primary biliary cholangitis',
        'CymaBay LIVDELZI {state} formulary coverage adoption',
    ],
}

# Fields with expected value types (used in extraction prompt)
FIELD_SCHEMAS = {
    "pbc_patients_adults": {
        "type": "integer",
        "description": "Estimated number of adult PBC patients in the state",
        "example": 2500,
    },
    "second_line_eligible": {
        "type": "integer",
        "description": "Estimated adults eligible for 2nd-line therapy (UDCA non-responders, ~30% of total)",
        "example": 700,
    },
    "hepatology_density": {
        "type": "object",
        "description": "Hepatologist density — label (low/medium/high) and specialists per million",
        "example": {"label": "low", "per_million": 9},
    },
    "medicare_pct": {
        "type": "number",
        "description": "Percentage of state adult population covered by Medicare (0-100)",
        "example": 17.0,
    },
    "dominant_payer": {
        "type": "object",
        "description": "Largest commercial insurer in the state with estimated market share",
        "example": {"insurer": "Cigna", "share_pct": 40},
    },
    "second_line_share_lean": {
        "type": "string",
        "description": "2L prescribing lean: 'parity', 'IQIRVO-lean', 'LIVDELZI-lean', or 'Ocaliva-lean'",
        "example": "parity",
    },
    "ex_ocaliva": {
        "type": "string",
        "description": "Status of ex-Ocaliva patients in the state (switched, remaining, unknown)",
        "example": "mostly switched",
    },
    "iqirvo_share": {
        "type": "number",
        "description": "IQIRVO estimated market share % among 2L PBC patients in state",
        "example": 35.0,
    },
    "livdelzi_share": {
        "type": "number",
        "description": "LIVDELZI estimated market share % among 2L PBC patients in state",
        "example": 30.0,
    },
}
