"""
CRRA Lab C4 — Portfolio Orchestrator

Runs the whole renewal review as a LangGraph StateGraph:

    analysis  ->  policy_check  -+-> hitl -> report -> END
                                 +--------> report -> END   (no policy trigger)

The HITL gate is the point of this lab. An agent may recommend, but under the
procurement policy it may not commit Zensar to anything on its own.

Prerequisites:
    1. python data/kb_setup.py                (loads the policy KB)
    2. python mcp_server/contract_shim.py     (second terminal, port 5001)

Run from the project root:
    python orchestrator/supervisor.py                     # default portfolio
    python orchestrator/supervisor.py CTR-1005 CTR-1004   # specific contracts
"""

import sys
from pathlib import Path

# Running `python orchestrator/supervisor.py` puts only orchestrator/ on the
# import path, not the project root, so guardrails/ and data/ would not be
# found. Add the project root BEFORE any project-local import.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
from typing import TypedDict  # noqa: E402

# Windows consoles can choke on the box-drawing characters used below.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ══════════════════════════════════════════════════════════════
# DEPENDENCIES — install into THIS interpreter if missing
# ══════════════════════════════════════════════════════════════

REQUIRED_PACKAGES = {  # import name -> pip package
    "anthropic": "anthropic",
    "chromadb": "chromadb",
    "requests": "requests",
    "dotenv": "python-dotenv",
    "langgraph": "langgraph",
}


def _ensure_packages() -> None:
    missing = []
    for module, package in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            missing.append(package)
    if not missing:
        return
    print(f"Installing missing packages into {sys.executable}: {', '.join(missing)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    except (subprocess.CalledProcessError, OSError) as exc:
        raise SystemExit(
            f"\nCould not install {', '.join(missing)} ({exc}).\n"
            f'Install manually:  & "{sys.executable}" -m pip install {" ".join(missing)}'
        )
    importlib.invalidate_caches()


_ensure_packages()

import anthropic  # noqa: E402
import chromadb  # noqa: E402
import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402

from guardrails.audit_logger import AuditLogger  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

MODEL = "claude-opus-5"
OUTPUT_CONFIG = {"effort": "medium"}  # temperature is NOT supported on this model
MAX_TOKENS = 4096                     # room for thinking + the tool call
CONTRACT_API = "http://localhost:5001"
KB_COLLECTION = "crra_policy"
MAX_ROUNDS = 5

# CTR-1010 has no policy trigger and should skip the gate entirely — that
# contrast is the point. CTR-1012 and CTR-1006 must stop for a human.
DEFAULT_PORTFOLIO = ["CTR-1010", "CTR-1012", "CTR-1006"]

audit = AuditLogger()
_client = None
_kb = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"].strip())
    return _client


def create_message(**kwargs):
    """messages.create with output_config.

    requirements.txt pins anthropic==0.40.0, which predates the output_config
    parameter and would raise TypeError. On such versions send it as a raw body
    field instead — the API receives exactly the same request.
    """
    create = get_client().messages.create
    if "output_config" in inspect.signature(create).parameters:
        return create(output_config=OUTPUT_CONFIG, **kwargs)
    return create(extra_body={"output_config": OUTPUT_CONFIG}, **kwargs)


def extract_text(response) -> str:
    """First block with a .text attribute — a thinking block may come first."""
    for block in response.content:
        if hasattr(block, "text"):
            return block.text.strip()
    return ""


# ══════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════

class ContractState(TypedDict):
    contract_id: str
    contract: dict

    recommendation: str
    confidence: str
    rationale: str
    policy_citation: str
    estimated_annual_impact_inr: int

    hitl_required: bool
    hitl_reason: str
    hitl_approved: bool
    approver: str

    final_status: str


def new_state(contract_id: str) -> ContractState:
    return {
        "contract_id": contract_id, "contract": {}, "recommendation": "", "confidence": "",
        "rationale": "", "policy_citation": "", "estimated_annual_impact_inr": 0,
        "hitl_required": False, "hitl_reason": "", "hitl_approved": False,
        "approver": "", "final_status": "",
    }


# ══════════════════════════════════════════════════════════════
# POLICY KB  (Lab C1)
# ══════════════════════════════════════════════════════════════

def get_kb():
    """chromadb.Client() is in-memory, so build the KB from data/kb/ if absent."""
    global _kb
    if _kb is not None:
        return _kb
    kb_client = chromadb.Client()
    try:
        _kb = kb_client.get_collection(KB_COLLECTION)
    except Exception:
        from data.kb_setup import chunk_article

        _kb = kb_client.create_collection(KB_COLLECTION)
        ids, docs, metas = [], [], []
        for md in sorted((PROJECT_ROOT / "data" / "kb").glob("*.md")):
            for c in chunk_article(md.read_text(encoding="utf-8"), md.name):
                ids.append(c["id"])
                docs.append(c["document"])
                metas.append(c["metadata"])
        _kb.add(ids=ids, documents=docs, metadatas=metas)
        print(f"  (built policy KB: {len(ids)} chunks)")
    return _kb


def search_policy(query: str) -> list[dict]:
    """Best section per source file, for the top 2 files."""
    res = get_kb().query(query_texts=[query], n_results=6)
    best: dict[str, tuple[float, str, str]] = {}
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        src = meta.get("source", "unknown")
        if src not in best or dist < best[src][0]:
            best[src] = (dist, meta.get("heading", "(no heading)"), doc)
    ranked = sorted(best.items(), key=lambda kv: kv[1][0])[:2]
    return [
        {"source": s, "section": h, "confidence": round(1 - d, 2), "text": t}
        for s, (d, h, t) in ranked
    ]


# ══════════════════════════════════════════════════════════════
# NODE 1 — ANALYSIS
# ══════════════════════════════════════════════════════════════

ANALYSIS_TOOLS = [
    {
        "name": "search_policy",
        "description": (
            "Search the BizOps procurement policy KB. Returns the best section from "
            "each of the top 2 policy files. Call at most twice."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Plain-English policy question"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "submit_recommendation",
        "description": "Record the final recommendation. Call exactly once, last.",
        "input_schema": {
            "type": "object",
            "properties": {
                "recommendation": {
                    "type": "string",
                    "enum": ["RENEW", "RENEGOTIATE", "CONSOLIDATE", "TERMINATE"],
                },
                "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                "rationale": {
                    "type": "string",
                    "description": "Two or three sentences stating the numbers that drove the decision.",
                },
                "policy_citation": {
                    "type": "string",
                    "description": "Policy file and section, e.g. 'auto_renewal_rules.md §Escalation trigger'",
                },
                "estimated_annual_impact_inr": {
                    "type": "integer",
                    "description": "Rough rupee impact per year. 0 for RENEW at existing terms. Negative = saving.",
                },
            },
            "required": ["recommendation", "confidence", "rationale", "policy_citation"],
        },
    },
]

ANALYSIS_SYSTEM = """You are the Renewal Analysis Agent for Zensar BizOps.

Decide one of RENEW, RENEGOTIATE, CONSOLIDATE, TERMINATE for the contract given.
Use only the numbers in the contract record — never invent them.

Search the policy KB (at most twice) to find the governing rule, then call
submit_recommendation exactly once.

Guidance:
- Utilisation under 40% with an overlapping vendor in the same category -> CONSOLIDATE.
- An unassigned owner is not a licence to TERMINATE — nobody has confirmed the
  capability is unneeded, so recommend conservatively and say why.
- Proposed uplift above 15% -> RENEGOTIATE, never accept at first offer.
- Healthy utilisation plus modest uplift -> RENEW.
- TERMINATE only where the capability itself is no longer required.

LOW confidence is a valid, useful answer. Say so plainly instead of inventing
certainty, and do not keep searching hoping for a cleaner picture.

You do not decide whether a human must approve — a separate policy check does.
Always cite the specific policy file and section that justifies your call."""

CONTRACT_FACTS = (
    "contract_id", "vendor", "category", "owner", "annual_value_inr",
    "approval_band", "renewal_date", "notice_state", "auto_renew",
    "seats_purchased", "seats_active", "utilisation_pct", "proposed_uplift_pct",
)


def _fallback(state: ContractState, contract: dict, why: str) -> ContractState:
    """No recommendation submitted: record a conservative LOW so a human sees it."""
    return {
        **state, "contract": contract, "recommendation": "RENEGOTIATE",
        "confidence": "LOW", "rationale": why,
        "policy_citation": "n/a", "estimated_annual_impact_inr": 0,
    }


def analysis_node(state: ContractState) -> ContractState:
    cid = state["contract_id"]
    print(f"\n{'═' * 66}\nCONTRACT: {cid}\n{'═' * 66}")
    print("\n▶ ANALYSIS AGENT")

    # ---- 1. Fetch the contract ------------------------------------------
    try:
        r = requests.get(f"{CONTRACT_API}/api/contracts/{cid}", timeout=10)
        contract = r.json()
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Contract API unreachable: {e}")
        print("     Is mcp_server/contract_shim.py running on port 5001?")
        audit.log("AnalysisAgent", "contract_fetch_failed", cid, f"API unreachable: {e}")
        return {**state, "final_status": "ERROR_API_UNREACHABLE", "hitl_required": False}
    except ValueError:
        audit.log("AnalysisAgent", "contract_fetch_failed", cid, "API returned non-JSON")
        return {**state, "final_status": "ERROR_BAD_RESPONSE", "hitl_required": False}

    if "error" in contract:
        print(f"  ✗ {contract['error']}")
        audit.log("AnalysisAgent", "contract_fetch_failed", cid, contract["error"])
        return {**state, "final_status": "ERROR_NOT_FOUND", "hitl_required": False}

    util = contract.get("utilisation_pct")
    print(
        f"  {contract.get('vendor')} · {contract.get('category')} · band {contract.get('approval_band')}\n"
        f"  INR {contract.get('annual_value_inr', 0):,}/yr · "
        f"util {'n/a' if util is None else f'{util}%'} · "
        f"uplift {contract.get('proposed_uplift_pct')}% · {contract.get('notice_state')}"
    )
    audit.log(
        "AnalysisAgent", "contract_fetched", cid,
        f"{contract.get('vendor')} band {contract.get('approval_band')} "
        f"{contract.get('notice_state')} owner {contract.get('owner')}",
    )

    # ---- 2. Tool-calling loop, capped at MAX_ROUNDS ---------------------
    facts = json.dumps({k: contract.get(k) for k in CONTRACT_FACTS}, indent=2)
    messages = [{"role": "user", "content": f"Analyse this contract:\n{facts}"}]

    for rnd in range(1, MAX_ROUNDS + 1):
        response = create_message(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=ANALYSIS_SYSTEM,
            tools=ANALYSIS_TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            text = extract_text(response)
            print(f"  Agent finished without a recommendation ({response.stop_reason}): {text[:150]}")
            audit.log("AnalysisAgent", "analysis_incomplete", cid,
                      f"Stopped ({response.stop_reason}) without submitting: {text[:200]}")
            return _fallback(state, contract, text[:300] or "No recommendation produced; needs manual review.")

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []

        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue

            if block.name == "submit_recommendation":
                a = block.input
                print(f"  → {a['recommendation']} (confidence {a['confidence']})")
                audit.log(
                    "AnalysisAgent", "recommendation", cid,
                    f"{a['recommendation']} / {a['confidence']} — {a['policy_citation']}",
                    round=rnd,
                )
                return {
                    **state,
                    "contract": contract,
                    "recommendation": a["recommendation"],
                    "confidence": a["confidence"],
                    "rationale": a.get("rationale", ""),
                    "policy_citation": a.get("policy_citation", ""),
                    "estimated_annual_impact_inr": a.get("estimated_annual_impact_inr", 0) or 0,
                }

            if block.name == "search_policy":
                query = block.input.get("query", "")
                try:
                    hits = search_policy(query)
                    tops = ", ".join(f"{h['source']} §{h['section']} ({h['confidence']:.0%})" for h in hits)
                    print(f'  → policy: "{query[:40]}" -> {tops or "no results"}')
                    audit.log("AnalysisAgent", "policy_search", cid, f'"{query[:40]}" -> {tops}')
                    result = {"results": hits}
                except Exception as e:
                    print(f"  ✗ policy search failed: {e}")
                    audit.log("AnalysisAgent", "policy_search_failed", cid, str(e))
                    result = {"error": f"Policy KB unavailable: {e}", "results": []}
            else:
                result = {"error": f"unknown tool {block.name}"}

            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)}
            )

        messages.append({"role": "user", "content": tool_results})

    print(f"  ⚠️  No recommendation after {MAX_ROUNDS} rounds — recording LOW confidence.")
    audit.log("AnalysisAgent", "analysis_incomplete", cid, f"Stopped after {MAX_ROUNDS} rounds")
    return _fallback(state, contract, "Analysis did not converge; needs manual review.")


# ══════════════════════════════════════════════════════════════
# NODE 2 — POLICY CHECK  (pure Python — no model call)
# ══════════════════════════════════════════════════════════════

def policy_check_node(state: ContractState) -> ContractState:
    print("\n▶ POLICY CHECK")
    cid = state["contract_id"]
    contract = state.get("contract") or {}

    if not contract:  # analysis failed to fetch; report will record the error
        print("  Skipped — no contract data")
        audit.log("PolicyCheck", "evaluate_triggers", cid,
                  f"skipped — {state.get('final_status') or 'no contract data'}")
        return {**state, "hitl_required": False, "hitl_reason": ""}

    # Every matching trigger is collected — never stop at the first one.
    reasons: list[str] = []

    band = str(contract.get("approval_band", "")).strip().upper()
    if band in ("B", "C"):
        reasons.append(f"approval band {band} requires a named human approver")

    if str(contract.get("notice_state", "")).strip().upper() == "INSIDE_WINDOW":
        reasons.append("inside the notice window — leverage already lost")

    if str(state.get("recommendation", "")).upper() == "TERMINATE":
        reasons.append("all terminations require written owner confirmation")

    if str(state.get("confidence", "")).upper() == "LOW":
        reasons.append("analysis confidence is LOW")

    if str(contract.get("owner", "")).strip().upper() == "UNASSIGNED":
        reasons.append("no business owner on record")

    hitl_required = bool(reasons)
    reason_text = "; ".join(reasons) if reasons else "no policy trigger"

    print(f"  HITL required: {hitl_required}")
    for r in reasons:
        print(f"    · {r}")

    audit.log("PolicyCheck", "evaluate_triggers", cid, reason_text,
              "PENDING" if hitl_required else "N/A", triggers=len(reasons))

    return {**state, "hitl_required": hitl_required, "hitl_reason": reason_text}


def route_after_policy_check(state: ContractState) -> str:
    return "hitl" if state.get("hitl_required") else "report"


# ══════════════════════════════════════════════════════════════
# NODE 3 — HITL GATE
# ══════════════════════════════════════════════════════════════

def _ask(prompt: str) -> str | None:
    try:
        return input(prompt).strip()
    except EOFError:  # no console attached (piped or scheduled run)
        return None


def hitl_node(state: ContractState) -> ContractState:
    cid = state["contract_id"]
    impact = state.get("estimated_annual_impact_inr") or 0

    print("\n▶ HUMAN APPROVAL GATE")
    print(f"  Contract : {cid}  ({state['contract'].get('vendor')})")
    print(f"  Proposed : {state['recommendation']}  (confidence {state['confidence']})")
    print(f"  Policy   : {state['policy_citation']}")
    if impact:
        print(f"  Impact   : INR {impact:,}/yr")
    print("  Because  :")
    for reason in state["hitl_reason"].split("; "):
        print(f"    · {reason}")
    print(f"\n  {state['rationale']}")

    audit.log("HITLGate", "approval_request", cid,
              f"{state['recommendation']} — {state['hitl_reason']}", "PENDING")

    answer = None
    while answer not in ("y", "n"):
        raw = _ask("\n  Approve this recommendation? [y/n]: ")
        if raw is None:
            print("  No console input available — treating as REJECTED.")
            answer = "n"
        else:
            answer = raw.lower()[:1]
            if answer not in ("y", "n"):
                print("  Please type y or n.")

    approver = ""
    if answer == "y":
        while not approver:
            raw = _ask("  Approver name: ")
            if raw is None:
                break
            approver = raw
        if not approver:  # an approval nobody will put their name to does not count
            print("  No approver name given — treating as REJECTED.")
            answer = "n"

    approved = answer == "y"
    audit.log(
        "HITLGate", "approval_decision", cid,
        f"{state['recommendation']} {'approved' if approved else 'rejected'}",
        "APPROVED" if approved else "REJECTED",
        actor=approver if approved else "human reviewer",
    )
    print(f"  → {'APPROVED by ' + approver if approved else 'REJECTED — no action will be taken'}")

    return {**state, "hitl_approved": approved, "approver": approver}


# ══════════════════════════════════════════════════════════════
# NODE 4 — REPORT
# ══════════════════════════════════════════════════════════════

def report_node(state: ContractState) -> ContractState:
    print("\n▶ REPORT")
    cid = state["contract_id"]

    if state.get("final_status", "").startswith("ERROR"):
        print(f"  Skipped — {state['final_status']}")
        audit.log("Reporting", "final_status", cid, state["final_status"])
        return state

    rec = state["recommendation"]
    if not state.get("hitl_required"):
        final = f"{rec}_AUTO"
        note = "Actioned without human approval (no policy trigger)."
        status = "N/A"
    elif state.get("hitl_approved"):
        final = f"{rec}_APPROVED"
        note = f"Approved by {state.get('approver')}. Cleared to action."
        status = "APPROVED"
    else:
        final = "ON_HOLD_REJECTED"
        note = "Rejected at the approval gate. No commitment made to the vendor."
        status = "REJECTED"

    audit.log(
        "Reporting", "final_status", cid, f"{final} — {note}", status,
        actor=state.get("approver") or "system",
        final_status=final,
        recommendation=rec,
        confidence=state.get("confidence"),
        policy_citation=state.get("policy_citation"),
        estimated_annual_impact_inr=state.get("estimated_annual_impact_inr"),
        hitl_reason=state.get("hitl_reason"),
    )
    print(f"  FINAL STATUS: {final}")
    print(f"  {note}")

    return {**state, "final_status": final}


# ══════════════════════════════════════════════════════════════
# GRAPH
# ══════════════════════════════════════════════════════════════

def build_graph():
    g = StateGraph(ContractState)
    g.add_node("analysis", analysis_node)
    g.add_node("policy_check", policy_check_node)
    g.add_node("hitl", hitl_node)
    g.add_node("report", report_node)

    g.set_entry_point("analysis")
    g.add_edge("analysis", "policy_check")
    g.add_conditional_edges(
        "policy_check", route_after_policy_check, {"hitl": "hitl", "report": "report"}
    )
    g.add_edge("hitl", "report")
    g.add_edge("report", END)
    return g.compile()


def main(argv: list[str]) -> int:
    if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        raise SystemExit(
            "ANTHROPIC_API_KEY not set. Copy .env.template to .env and add your key,\n"
            'or run:  $env:ANTHROPIC_API_KEY = "sk-ant-..."'
        )

    graph = build_graph()
    portfolio = argv or DEFAULT_PORTFOLIO

    outcomes = []
    for cid in portfolio:
        try:
            outcomes.append(graph.invoke(new_state(cid)))
        except anthropic.APIError as e:
            print(f"  ✗ Anthropic API error: {e}")
            audit.log("Supervisor", "api_error", cid, str(e))
            outcomes.append({**new_state(cid), "final_status": "ERROR_ANTHROPIC_API"})

    print(f"\n\n{'═' * 72}\nPORTFOLIO REVIEW COMPLETE\n{'═' * 72}")
    print(f"{'Contract':<11}{'Action':<14}{'Conf':<7}{'Gate':<8}{'Approver':<17}{'Final'}")
    print("-" * 72)
    for o in outcomes:
        gate = "HUMAN" if o.get("hitl_required") else "auto"
        print(
            f"{o['contract_id']:<11}{(o.get('recommendation') or '—'):<14}"
            f"{(o.get('confidence') or '—'):<7}{gate:<8}"
            f"{(o.get('approver') or '—')[:16]:<17}{o.get('final_status') or '—'}"
        )

    audit.summary()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))