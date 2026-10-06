"""Fetch resolved tickets + full comment threads from Zendesk REST API.

Uses the Incremental Ticket Export API (cursor-based) for the full export —
the standard /tickets.json endpoint is capped at 10,000 results (100 pages).
All comments are fetched in full — no truncation. Cleaning happens downstream.
"""

import asyncio
import logging
import time
from typing import Generator
import httpx

logger = logging.getLogger(__name__)


class ZendeskExtractor:
    def __init__(self, subdomain: str, email: str, api_key: str):
        self.base_url = f"https://{subdomain}.zendesk.com/api/v2"
        self.auth = (f"{email}/token", api_key)
        self.client = httpx.Client(auth=self.auth, timeout=30)

    def _get(self, url: str, params: dict = None) -> dict:
        """GET with retry on rate limit."""
        for attempt in range(3):
            resp = self.client.get(url, params=params)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                logger.warning(f"Rate limited — waiting {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Failed after 3 attempts: {url}")

    def fetch_solved_tickets(self, updated_after: str = None) -> Generator[dict, None, None]:
        """
        Yield all solved tickets using the Incremental Export API (no 10k limit).

        For a full export, start_time=0 fetches from the beginning.
        For incremental runs, updated_after is an ISO8601 string that gets
        converted to a Unix timestamp.
        """
        if updated_after:
            from datetime import datetime, timezone
            try:
                dt = datetime.fromisoformat(updated_after.replace("Z", "+00:00"))
                start_ts = int(dt.timestamp())
            except Exception:
                start_ts = 0
        else:
            start_ts = 0  # Full export from the beginning

        url = f"{self.base_url}/incremental/tickets/cursor.json"
        params = {"start_time": start_ts}
        total_fetched = 0

        while url:
            data = self._get(url, params)
            tickets = data.get("tickets", [])

            for ticket in tickets:
                # Incremental export returns ALL statuses — filter to solved/closed only
                if ticket.get("status") in ("solved", "closed"):
                    yield ticket
                    total_fetched += 1

            after_cursor = data.get("after_cursor")
            end_of_stream = data.get("end_of_stream", False)

            if end_of_stream or not after_cursor:
                logger.info(f"End of stream — total fetched: {total_fetched}")
                break

            url = f"{self.base_url}/incremental/tickets/cursor.json"
            params = {"cursor": after_cursor}

            if tickets:
                logger.info(f"Fetched {total_fetched} solved tickets so far...")

    def fetch_comments(self, ticket_id: int) -> list[dict]:
        """Fetch all comments for a ticket — full thread, no truncation."""
        url = f"{self.base_url}/tickets/{ticket_id}/comments.json"
        data = self._get(url)
        return data.get("comments", [])

    def fetch_ticket_with_comments(self, ticket: dict) -> dict:
        """Enrich a ticket dict with its full comment thread."""
        ticket_id = ticket["id"]
        try:
            comments = self.fetch_comments(ticket_id)
            ticket["comments"] = comments
        except Exception as e:
            logger.warning(f"Could not fetch comments for ticket {ticket_id}: {e}")
            ticket["comments"] = []
        return ticket

    async def _afetch_comments(self, async_client: httpx.AsyncClient, ticket: dict) -> dict:
        """Async fetch comments for a single ticket."""
        ticket_id = ticket["id"]
        url = f"{self.base_url}/tickets/{ticket_id}/comments.json"
        for attempt in range(3):
            try:
                resp = await async_client.get(url)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited — waiting {wait}s")
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                ticket["comments"] = resp.json().get("comments", [])
                return ticket
            except Exception as e:
                if attempt == 2:
                    logger.warning(f"Could not fetch comments for ticket {ticket_id}: {e}")
                    ticket["comments"] = []
                    return ticket
                await asyncio.sleep(2)
        return ticket

    async def fetch_tickets_with_comments_batch(self, tickets: list[dict], concurrency: int = 20) -> list[dict]:
        """Fetch comments for a batch of tickets in parallel."""
        auth = (f"{self.auth[0]}", self.auth[1])
        semaphore = asyncio.Semaphore(concurrency)

        async def fetch_with_semaphore(client, ticket):
            async with semaphore:
                return await self._afetch_comments(client, ticket)

        async with httpx.AsyncClient(auth=auth, timeout=30) as client:
            return await asyncio.gather(*[fetch_with_semaphore(client, t) for t in tickets])

    def close(self):
        self.client.close()
