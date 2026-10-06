"""
Jira exporter for voice-related projects.

Fetches resolved/done issues from:
  VT  (Voice-Team)         ~9k issues
  VPT (VoiceProductTeam)   ~4.6k issues
  ZEN (Zentrunk/SIP)       ~4.3k issues
  SUP (Customer Issues)    ~2.8k issues
  SP  (Support-Team)       ~1.6k issues
  CAL (Call Insights)      ~2.5k issues
  CSDK (Client SDK)        ~2.2k issues

Filters: status in (Done, Resolved, Closed) — only issues with actual resolutions.
Extracts plain text from Atlassian Document Format (ADF) descriptions + comments.
Checkpoints to ingestion/jira_checkpoint.jsonl — safe to kill and resume.

Usage:
  .venv/bin/python -m ingestion.jira_exporter
"""

import json
import logging
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

INGESTION_DIR = Path(__file__).parent
CHECKPOINT_FILE = INGESTION_DIR / "jira_checkpoint.jsonl"

PROJECTS = ["VT", "VPT", "ZEN", "SUP", "SP", "CAL", "CSDK"]
DONE_STATUSES = ("Done", "Resolved", "Closed", "Fixed", "Complete", "Won't Fix", "Won't Do")

FIELDS = [
    "summary", "description", "status", "issuetype", "priority",
    "created", "updated", "resolutiondate", "resolution",
    "comment", "labels", "assignee", "reporter", "project",
]


# ── ADF → plain text ──────────────────────────────────────────────────────────

def adf_to_text(node) -> str:
    """Recursively extract plain text from Atlassian Document Format (ADF) node."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node

    node_type = node.get("type", "")
    texts = []

    # Text leaf
    if node_type == "text":
        return node.get("text", "")

    # Code block — include as-is with label
    if node_type == "codeBlock":
        code = "".join(adf_to_text(c) for c in node.get("content", []))
        return f"\n[code]\n{code}\n[/code]\n"

    # Heading — add newline
    if node_type in ("heading",):
        text = "".join(adf_to_text(c) for c in node.get("content", []))
        return f"\n{text}\n"

    # List items
    if node_type in ("listItem", "bulletList", "orderedList"):
        parts = []
        for child in node.get("content", []):
            t = adf_to_text(child).strip()
            if t:
                parts.append(f"- {t}" if node_type != "orderedList" else t)
        return "\n".join(parts) + "\n"

    # Table — flatten rows
    if node_type in ("table", "tableRow", "tableCell", "tableHeader"):
        return " | ".join(
            adf_to_text(c).strip()
            for c in node.get("content", [])
            if adf_to_text(c).strip()
        ) + "\n"

    # Mention / inline nodes
    if node_type == "mention":
        return node.get("attrs", {}).get("text", "@mention")

    if node_type == "emoji":
        return node.get("attrs", {}).get("text", "")

    if node_type == "hardBreak":
        return "\n"

    # Recurse into content
    for child in node.get("content", []):
        t = adf_to_text(child)
        if t:
            texts.append(t)

    return "\n".join(texts) if node_type in ("doc", "blockquote", "panel") else " ".join(texts)


def extract_comment_text(comment: dict) -> str:
    """Extract plain text from a Jira comment."""
    body = comment.get("body")
    if not body:
        return ""
    if isinstance(body, str):
        return body
    return adf_to_text(body).strip()


# ── Checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint() -> set[str]:
    """Return set of already-exported issue IDs."""
    done = set()
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    done.add(str(entry["id"]))
    logger.info(f"Checkpoint: {len(done)} issues already exported")
    return done


def append_checkpoint(issue: dict):
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(json.dumps(issue) + "\n")


# ── Jira API ──────────────────────────────────────────────────────────────────

def fetch_issues(client: httpx.Client, project: str, already_done: set[str]) -> int:
    """
    Fetch all resolved issues for a project. Appends new ones to checkpoint.
    Returns count of newly exported issues.
    """
    jql = (
        f'project={project} AND status in '
        f'("Done","Resolved","Closed","Fixed","Complete","Won\'t Fix","Won\'t Do")'
        f' ORDER BY updated DESC'
    )

    token = None
    page = 0
    new_count = 0

    while True:
        body = {"jql": jql, "maxResults": 100, "fields": FIELDS}
        if token:
            body["nextPageToken"] = token

        try:
            resp = client.post(
                f"{settings.atlassian_base_url}/rest/api/3/search/jql",
                json=body,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"[{project}] API error on page {page}: {e}")
            time.sleep(5)
            continue

        issues = data.get("issues", [])

        for raw in issues:
            issue_id = str(raw["id"])
            if issue_id in already_done:
                continue

            fields = raw.get("fields", {})
            desc_raw = fields.get("description")
            desc_text = adf_to_text(desc_raw).strip() if isinstance(desc_raw, dict) else (desc_raw or "")

            comments = []
            for c in fields.get("comment", {}).get("comments", []):
                text = extract_comment_text(c)
                if text:
                    comments.append({
                        "author": c.get("author", {}).get("displayName", ""),
                        "created": c.get("created", ""),
                        "text": text,
                    })

            issue = {
                "id": issue_id,
                "key": raw.get("key", ""),
                "project": project,
                "summary": fields.get("summary", ""),
                "description": desc_text,
                "status": fields.get("status", {}).get("name", ""),
                "issue_type": fields.get("issuetype", {}).get("name", ""),
                "priority": fields.get("priority", {}).get("name", ""),
                "resolution": fields.get("resolution", {}).get("name", "") if fields.get("resolution") else "",
                "labels": fields.get("labels", []),
                "created": fields.get("created", ""),
                "updated": fields.get("updated", ""),
                "resolutiondate": fields.get("resolutiondate", ""),
                "comments": comments,
                "url": f"{settings.atlassian_base_url}/browse/{raw.get('key','')}",
            }

            append_checkpoint(issue)
            already_done.add(issue_id)
            new_count += 1

        page += 1
        if page % 5 == 0:
            logger.info(f"[{project}] Page {page} — {new_count} new issues so far")

        if data.get("isLast", True):
            break
        token = data.get("nextPageToken")
        time.sleep(0.1)  # polite pacing

    return new_count


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info(f"Starting Jira export for projects: {PROJECTS}")
    logger.info(f"Checkpoint file: {CHECKPOINT_FILE}")

    client = httpx.Client(
        auth=(settings.atlassian_email, settings.atlassian_api_key),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        follow_redirects=True,
    )

    already_done = load_checkpoint()
    total_new = 0

    for project in PROJECTS:
        logger.info(f"Fetching [{project}]...")
        count = fetch_issues(client, project, already_done)
        logger.info(f"[{project}] done — {count} new issues exported")
        total_new += count

    client.close()

    total = sum(1 for _ in open(CHECKPOINT_FILE)) if CHECKPOINT_FILE.exists() else 0
    logger.info(f"Export complete — {total_new} new | {total} total in {CHECKPOINT_FILE.name}")


if __name__ == "__main__":
    main()
