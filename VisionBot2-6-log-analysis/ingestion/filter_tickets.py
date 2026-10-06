"""
Reusable ticket filter script.

Scans raw_tickets_*.jsonl files and outputs a filtered JSON file
for a given product's tag set and date range.

Usage:
  .venv/bin/python -m ingestion.filter_tickets --product sms
  .venv/bin/python -m ingestion.filter_tickets --product sip
  .venv/bin/python -m ingestion.filter_tickets --product sms --all-years
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INGESTION_DIR = Path(__file__).parent
RAW_FILES = [
    INGESTION_DIR / "raw_tickets_2022-01-01.jsonl",
    INGESTION_DIR / "raw_tickets_2025-01-01.jsonl",
]

PRODUCT_CONFIG = {
    "sms": {
        "tags": {
            "sms", "messaging", "mms", "10dlc", "a2p_10dlc", "a2p", "dlt",
            "sender_id", "short_code", "shortcode", "toll_free_sms", "tollfree_sms",
            "long_code", "message_delivery", "message_failed", "sms_delivery",
            "undelivered", "opt_out", "opt_in", "template", "ucc", "campaign_registration",
        },
        "output": "sms_tickets.json",
    },
    "sip": {
        "tags": {
            "sip_trunking", "zentrunk", "sip_trunk", "sip", "trunking",
            "zentrunk_registration", "sip_registration", "trunk",
        },
        "output": "sip_tickets.json",
    },
    "numbers": {
        "tags": {
            "phone_number", "phone_numbers", "did", "number_provisioning",
            "number_porting", "porting", "number_management", "toll_free",
            "tollfree", "short_code", "shortcode", "number_not_working",
            "caller_id", "inbound_number", "local_number", "number_purchase",
            "number_release", "number_compliance", "regulatory",
            "activation_nic",  # India compliance activation tickets
        },
        "output": "numbers_tickets.json",
    },
}


def filter_tickets(product: str, recent_only: bool = True) -> None:
    config = PRODUCT_CONFIG[product]
    tag_set = config["tags"]
    output_file = INGESTION_DIR / config["output"]

    tickets = []
    total_scanned = 0

    for fpath in RAW_FILES:
        if not fpath.exists():
            logger.warning(f"File not found, skipping: {fpath}")
            continue
        logger.info(f"Scanning {fpath.name}...")
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total_scanned += 1

                # Date filter
                if recent_only:
                    created = t.get("created_at", "")
                    if not (created.startswith("2025") or created.startswith("2026")):
                        continue

                tags = set(str(tg).lower() for tg in (t.get("tags") or []))
                if tags & tag_set:
                    tickets.append(t)

    logger.info(
        f"Scanned {total_scanned:,} tickets — "
        f"found {len(tickets):,} {product} tickets"
        f"{' (2025-26 only)' if recent_only else ' (all years)'}"
    )

    with open(output_file, "w") as f:
        json.dump(tickets, f)
    logger.info(f"Written to {output_file.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--product",
        required=True,
        choices=list(PRODUCT_CONFIG.keys()),
        help="Product to filter tickets for",
    )
    parser.add_argument(
        "--all-years",
        action="store_true",
        help="Include all years (default: 2025-26 only)",
    )
    args = parser.parse_args()
    filter_tickets(args.product, recent_only=not args.all_years)


if __name__ == "__main__":
    main()
