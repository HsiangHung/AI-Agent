"""
extractor.py — uses Claude to extract structured PBC fields from raw text.
Returns typed Python values with a confidence score + source citation.
"""

import json
import logging
import anthropic

from config import ANTHROPIC_API_KEY, FIELD_SCHEMAS, STATE_POPULATION

log = logging.getLogger(__name__)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# System prompt establishes the extraction persona
SYSTEM_PROMPT = """You are a pharmaceutical market analyst specializing in rare liver diseases.
Your task is to extract specific data fields about Primary Biliary Cholangitis (PBC) from
web-scraped text and search snippets. You must return ONLY valid JSON matching the requested schema.

Key domain knowledge:
- PBC (Primary Biliary Cholangitis) affects ~35-40 per 100,000 adults in the US
- ~30% of PBC patients are UDCA non-responders, making them 2nd-line (2L) eligible
- Hepatology density is classified: low (<15/M), medium (15-30/M), high (>30/M)
- Key 2L drugs: IQIRVO (elafibranor, Ipsen, approved 2024), LIVDELZI (seladelpar, CymaBay/GSK, approved 2024), Ocaliva (obeticholic acid, Intercept — FDA withdrew indication 2024)
- Medicare typically covers 17-22% of adults depending on state age distribution
- Dominant payers vary: BCBS, UnitedHealth, Cigna, Aetna, Humana depending on state

When data is not found in the context, use your domain knowledge to provide a reasoned estimate
and mark confidence as "estimated". Always prefer explicit data over estimates.
"""

EXTRACTION_PROMPT = """Extract the field "{field}" for the US state of {state} from the context below.

Field description: {description}
Expected type: {field_type}
Example value: {example}

Context:
{context}

Return a JSON object with exactly these keys:
{{
  "value": <extracted or estimated value matching the expected type>,
  "confidence": "high" | "medium" | "low" | "estimated",
  "source": "<URL or 'domain knowledge estimate' or 'calculated from X'>",
  "notes": "<brief explanation or caveat>"
}}

If extracting a number and only a national statistic is available, scale it proportionally by
state population ({state_pop:,} for {state}, US total ~330M).

Return only the JSON object, no other text.
"""


def extract_field(
    field: str,
    state: str,
    context: str,
) -> dict:
    """
    Call Claude to extract one PBC field for one state from scraped context.
    Returns a dict: {value, confidence, source, notes}
    """
    schema = FIELD_SCHEMAS[field]
    state_pop = STATE_POPULATION.get(state, 5_000_000)

    prompt = EXTRACTION_PROMPT.format(
        field       = field,
        state       = state,
        description = schema["description"],
        field_type  = schema["type"],
        example     = json.dumps(schema["example"]),
        context     = context[:6000],  # stay within context budget
        state_pop   = state_pop,
    )

    try:
        message = client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = 512,
            system     = SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)
        log.info("Extracted %s for %s: %s (conf=%s)", field, state,
                 result.get("value"), result.get("confidence"))
        return result

    except json.JSONDecodeError as e:
        log.error("JSON parse error for %s/%s: %s\nRaw: %s", field, state, e, raw[:200])
        return {"value": None, "confidence": "failed", "source": "parse_error", "notes": str(e)}
    except Exception as e:
        log.error("Extraction failed for %s/%s: %s", field, state, e)
        return {"value": None, "confidence": "failed", "source": "api_error", "notes": str(e)}


def batch_extract_all_fields(state: str, field_contexts: dict[str, str]) -> dict:
    """
    Extract all 9 PBC fields for a single state in one pass.
    field_contexts: {field_name -> scraped_context_text}
    Returns: {field_name -> extraction_result_dict}
    """
    results = {}
    for field, context in field_contexts.items():
        results[field] = extract_field(field, state, context)
    return results
