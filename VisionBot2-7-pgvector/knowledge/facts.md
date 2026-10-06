# Plivo Support — Known Facts

These are specific facts that don't appear in tickets or docs but are essential for correct answers.
Add new facts here as they are discovered. Each fact should be 1-3 sentences, precise, actionable.

---

## Voice / CPS

**CPS = -1 means voice is intentionally blocked.**
A `CPS Allowed` value of -1 is NOT a rate limit — it means voice has been deliberately disabled at the account level, typically for fraud control or policy enforcement. It is set by Product/Ops/Engineering and cannot be changed via the Ryuk UI (Ryuk does not accept -1 as input). Relevant Jira refs: SUP-2627, VT-4212, VT-5752, VT-4427, VT-3677.

---

## Numbers / Routing

**EU/Irish local numbers require domestic routing to be provisioned.**
When a customer reports calls failing on a local Irish or EU number, check first if domestic routing is provisioned for their account. If yes: make the necessary routing changes. If no: advise international caller ID as a workaround — do not simply tell them to change their caller ID without checking routing first.

---

## Carrier Escalation

**Jio escalation requires "circuit ID" or "10-digit Service ID".**
When escalating a trunk or routing issue to Jio, Jio support asks for the "circuit ID" or "10-digit Service ID" — this is Jio's name for the trunk/circuit identifier in their system. Agents should provide this when opening a Jio carrier ticket.

---

## MPC / Conference Calls

**MPC (Multi-Party Conference) choppy audio is typically a codec or bridge capacity issue.**
Common patterns: choppy/robotic audio in conference calls (not one-to-one), degradation when more participants join, audio cutting after a bridge timeout. Check Ryuk Voice CDRs for the bridge leg specifically — the issue often appears only on the bridge-to-participant leg, not the originating leg.
