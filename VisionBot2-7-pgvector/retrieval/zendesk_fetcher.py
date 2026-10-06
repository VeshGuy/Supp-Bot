"""
Fetch a single Zendesk ticket + all comments + attachments by ticket ID.
Used when an agent pastes a Zendesk URL or bare ticket ID into the chat.

Resolves all user IDs (assignee, requester, comment authors) to real names
via a single batch /users/show_many call.

Attachments:
- Text/log files (.txt, .log, .har, .json, .csv, .xml) under 100KB: fetched and included inline.
- Images/PDFs: filename noted but content not extracted (would require vision API).
"""

import logging
import asyncio
import httpx

logger = logging.getLogger(__name__)

# Text-based content types we can fetch and include inline
_TEXT_CONTENT_TYPES = {
    "text/plain", "text/csv", "text/xml", "application/json",
    "application/xml", "application/octet-stream",
}
_TEXT_EXTENSIONS = {".txt", ".log", ".har", ".json", ".csv", ".xml", ".sip", ".pcap_text"}
_MAX_ATTACHMENT_BYTES = 100_000  # 100KB per attachment


def _is_text_attachment(filename: str, content_type: str) -> bool:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in _TEXT_EXTENSIONS or content_type in _TEXT_CONTENT_TYPES


async def _fetch_attachment_text(client: httpx.AsyncClient, url: str, filename: str, size: int) -> str:
    """Fetch a text attachment and return its content (truncated if large)."""
    if size > _MAX_ATTACHMENT_BYTES:
        return f"[truncated — file is {size:,} bytes, showing first 100KB]\n"
    try:
        resp = await client.get(url, timeout=15)
        resp.raise_for_status()
        text = resp.text[:_MAX_ATTACHMENT_BYTES]
        return text
    except Exception as e:
        logger.warning(f"Could not fetch attachment {filename}: {e}")
        return f"[could not fetch: {e}]"


async def fetch_ticket_by_id(ticket_id: str, subdomain: str, email: str, api_key: str) -> str:
    """
    Fetch ticket details + all comments + attachments from Zendesk API.
    Returns formatted plain text ready for injection into the LLM context.
    """
    base_url = f"https://{subdomain}.zendesk.com/api/v2"
    auth = (f"{email}/token", api_key)

    async with httpx.AsyncClient(auth=auth, timeout=20) as client:
        # Fetch ticket + comments in parallel
        ticket_resp, comments_resp = await asyncio.gather(
            client.get(f"{base_url}/tickets/{ticket_id}.json"),
            client.get(f"{base_url}/tickets/{ticket_id}/comments.json"),
        )
        ticket_resp.raise_for_status()
        comments_resp.raise_for_status()

        ticket = ticket_resp.json()["ticket"]
        comments = comments_resp.json().get("comments", [])

        # Collect all user IDs to resolve in one batch call
        user_ids = set()
        if ticket.get("assignee_id"):
            user_ids.add(ticket["assignee_id"])
        if ticket.get("requester_id"):
            user_ids.add(ticket["requester_id"])
        for c in comments:
            if c.get("author_id"):
                user_ids.add(c["author_id"])

        # Batch-resolve all user IDs → names
        user_map = {}
        if user_ids:
            ids_param = ",".join(str(i) for i in user_ids)
            try:
                users_resp = await client.get(
                    f"{base_url}/users/show_many.json",
                    params={"ids": ids_param},
                )
                if users_resp.status_code == 200:
                    for u in users_resp.json().get("users", []):
                        user_map[u["id"]] = u.get("name", f"User {u['id']}")
            except Exception as e:
                logger.warning(f"Failed to resolve user names: {e}")

        # Pre-fetch text attachments for all comments
        attachment_contents: dict[str, str] = {}  # attachment_id → content
        for c in comments:
            for att in c.get("attachments", []):
                att_id = str(att.get("id", ""))
                fname = att.get("file_name", "")
                ctype = att.get("content_type", "")
                size = att.get("size", 0)
                if _is_text_attachment(fname, ctype):
                    url = att.get("content_url", "")
                    if url:
                        content = await _fetch_attachment_text(client, url, fname, size)
                        attachment_contents[att_id] = content
                        logger.info(f"Fetched text attachment: {fname} ({size:,} bytes)")
                else:
                    logger.debug(f"Skipping non-text attachment: {fname} ({ctype})")

    def _name(user_id) -> str:
        return user_map.get(user_id, f"User #{user_id}") if user_id else "Unassigned"

    subject    = ticket.get("subject", "N/A")
    status     = ticket.get("status", "N/A")
    priority   = ticket.get("priority") or "N/A"
    created    = ticket.get("created_at", "N/A")
    updated    = ticket.get("updated_at", "N/A")
    tags       = ", ".join(ticket.get("tags", [])) or "none"
    assignee   = _name(ticket.get("assignee_id"))
    requester  = _name(ticket.get("requester_id"))

    lines = [
        f"=== LIVE TICKET #{ticket_id} ===",
        f"Subject:   {subject}",
        f"Status:    {status}",
        f"Priority:  {priority}",
        f"Assignee:  {assignee}",
        f"Requester: {requester}",
        f"Created:   {created}",
        f"Updated:   {updated}",
        f"Tags:      {tags}",
        "",
        "--- Thread ---",
    ]

    for c in comments:
        visibility = "Public" if c.get("public") else "Internal"
        author = _name(c.get("author_id"))
        body = (c.get("plain_body") or c.get("body") or "").strip()
        lines.append(f"[{visibility} | {author} | {c.get('created_at', '')}]")
        lines.append(body)

        # Append attachments for this comment
        for att in c.get("attachments", []):
            att_id = str(att.get("id", ""))
            fname = att.get("file_name", "")
            ctype = att.get("content_type", "")
            size = att.get("size", 0)
            if att_id in attachment_contents:
                lines.append(f"\n[Attachment: {fname} ({size:,} bytes)]\n{attachment_contents[att_id]}\n[End attachment]")
            else:
                # Image, PDF, or other non-text file — note it exists
                lines.append(f"[Attachment: {fname} ({ctype}, {size:,} bytes) — image/binary, not extracted]")

        lines.append("")

    return "\n".join(lines)
