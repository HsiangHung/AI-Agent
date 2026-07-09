"""
main.py — orchestrates the PBC state data agent.

Usage:
  python main.py                              # all 50 states × 9 fields
  python main.py --states CA TX NY            # specific states (name or abbr)
  python main.py --fields pbc_patients_adults hepatology_density
  python main.py --resume                     # skip already-populated rows
  python main.py --states CA --verbose
"""

import argparse
import logging
import sys
import uuid

from config import US_STATES, STATE_ABBR, SEARCH_QUERIES, FIELD_SCHEMAS
from database import init_db, upsert_field, log_error, get_connection
from extractor import extract_field
from scraper import gather_context
from search_agent import search


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def has_data(state: str, field: str) -> bool:
    conn = get_connection()
    row = conn.execute(
        "SELECT confidence FROM pbc_state_data WHERE state=? AND field=?",
        (state, field),
    ).fetchone()
    conn.close()
    return row is not None and row["confidence"] != "failed"


def collect_context(state: str, field: str) -> str:
    """Try each query template in order; return first non-empty context."""
    for tmpl in SEARCH_QUERIES[field]:
        query = tmpl.format(state=state)
        results = search(query)
        if results:
            ctx = gather_context(results)
            if ctx:
                return ctx
    return ""


def run_one(state: str, field: str, run_id: str, log: logging.Logger) -> str:
    """Search → scrape → extract → save for one (state, field). Returns 'ok' or 'failed'."""
    abbr = STATE_ABBR[state]
    try:
        context = collect_context(state, field)
        if not context:
            log.warning("[%s] %-30s — no content found", abbr, field)
            log_error(state, field, "no_content", run_id)
            return "failed"

        result = extract_field(field, state, context)
        upsert_field(state, abbr, field, result, run_id)

        conf = result.get("confidence", "?")
        val  = result.get("value")
        log.info("[%s] %-30s → %-40s (conf=%s)", abbr, field, str(val)[:40], conf)
        return "ok" if conf != "failed" else "failed"

    except Exception as e:
        log.error("[%s] %s — unexpected error: %s", abbr, field, e)
        log_error(state, field, str(e), run_id)
        return "failed"


def resolve_states(raw: list[str]) -> list[str]:
    abbr_to_name = {v: k for k, v in STATE_ABBR.items()}
    states = []
    for s in raw:
        if s in US_STATES:
            states.append(s)
        elif s.upper() in abbr_to_name:
            states.append(abbr_to_name[s.upper()])
        else:
            logging.getLogger("main").warning("Unknown state %r — skipping", s)
    return states


def main():
    parser = argparse.ArgumentParser(
        description="PBC state data agent — search, scrape, extract, store."
    )
    parser.add_argument(
        "--states", nargs="+", metavar="STATE",
        help="State names or abbreviations to process (default: all 50).",
    )
    parser.add_argument(
        "--fields", nargs="+", metavar="FIELD",
        choices=list(FIELD_SCHEMAS.keys()),
        help="Fields to extract (default: all 9).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip (state, field) pairs that already have non-failed data.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    log = logging.getLogger("main")

    states = resolve_states(args.states) if args.states else list(US_STATES)
    if not states:
        log.error("No valid states specified.")
        sys.exit(1)

    fields  = args.fields or list(FIELD_SCHEMAS.keys())
    run_id  = uuid.uuid4().hex
    total   = len(states) * len(fields)

    init_db()
    log.info(
        "Run %s | %d state(s) × %d field(s) = %d tasks%s",
        run_id[:8], len(states), len(fields), total,
        " [resume mode]" if args.resume else "",
    )

    counts = {"ok": 0, "failed": 0, "skipped": 0}
    done   = 0

    for i, state in enumerate(states, 1):
        abbr = STATE_ABBR[state]
        log.info("── %d/%d  %s (%s) ──", i, len(states), state, abbr)

        for field in fields:
            done += 1

            if args.resume and has_data(state, field):
                log.debug("[%s] %s — already populated, skipping", abbr, field)
                counts["skipped"] += 1
                continue

            status = run_one(state, field, run_id, log)
            counts[status] += 1

    log.info(
        "Finished — ok=%d  failed=%d  skipped=%d  run_id=%s",
        counts["ok"], counts["failed"], counts["skipped"], run_id[:8],
    )
    if counts["failed"]:
        log.info("Tip: re-run with --resume to retry only failed entries.")


if __name__ == "__main__":
    main()