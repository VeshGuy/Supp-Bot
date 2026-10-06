# SupportIQ — Deferred Edge Cases

Edge cases observed in real usage that are not urgent enough to fix now but worth tracking.

---

## EC-001 — "could you give me a scenario where this case would be applicable?"

**Observed:** March 27, 2026 — Slack thread about WebRTC → PSTN fallback.

**What happened:** After a `find-similar` sequence where the bot correctly found no tickets/docs for the scenario, the agent asked "could you give me a scenario where this case would be applicable?" The bot responded with the standard `explain` 3-section format (What it is / How it works / Key constraints) — technically correct but not what was asked.

**What should have happened:** The agent wanted a concrete, real-world example scenario — e.g., "An agent using a WebRTC browser client loses connectivity mid-call due to corporate firewall blocking STUN/TURN. With fallback configured, the call would re-establish over PSTN to the same destination number." That's a scenario, not a definition.

**Why it happened:** "give me a scenario where X is applicable" doesn't map cleanly to any current intent. `explain` is the closest match but produces a definition-style answer. There's no `example` or `use-case` intent.

**Potential fix:** Add an `example` sub-intent under `explain`, or detect "scenario" / "use case" / "when would this apply" phrasing and instruct the model to answer with a concrete narrative example instead of a definition.

**Priority:** Low — users can rephrase to "give me an example of a webrtc call failing and falling back to pstn" and get a better answer.

---
