"""
CRRA Lab C3 — Renewal Analysis Agent

Given a contract, decides RENEW / RENEGOTIATE / CONSOLIDATE / TERMINATE, with a
confidence level and a citation to the procurement policy that justifies it.

Prerequisites:
    1. python data/kb_setup.py                  (Lab C1 — loads the policy KB)
    2. python mcp_server/contract_shim.py       (Lab C2 — in a second terminal)

Run from the project root:
    python agents/renewal_agent.py

Missing packages (anthropic, chromadb, requests, python-dotenv) are installed
automatically into the Python that is running this script, so the
"No module named 'anthropic'" error cannot come from a venv mismatch.
"""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

# Windows consoles can choke on the box-drawing characters used below.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ══════════════════════════════════════════════════════════════
# DEPENDENCIES — install into THIS interpreter if missing
# ══════════════════════════════════════════════════════════════

# import name -> pip package name
REQUIRED_PACKAGES = {
    "anthropic": "anthropic",
    "chromadb": "chromadb",
    "requests": "requests",
    "dotenv": "python-dotenv",
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
            f"Install them manually into this Python, then run again:\n"
            f'  & "{sys.executable}" -m pip install {" ".join(missing)}\n'
            "If chromadb fails to build, recreate the venv on Python 3.12:\n"
            "  py -3.12 -m venv labenv"
        )
    importlib.invalidate_caches()


_ensure_packages()

import anthropic  # noqa: E402
import chromadb  # noqa: E402
import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

MODEL = "claude-opus-5"
CONTRACT_API = "http://localhost:5001"
KB_COLLECTION = "crra_policy"
MAX_TOKENS = 4096  # room for thinking + tool call; 1500 could truncate at effort=medium

# Hard cap so a model that will not converge cannot loop forever
MAX_ROUNDS = 5

client = None  # created in main() after the API key is checked
KB = None      # loaded lazily on the first search_policy call


# ══════════════════════════════════════════════════════════════
# TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════

def extract_text(response) -> str:
    """Return the first text block from a response.

    Newer models may put a thinking block first, so response.content[0].text is
    not safe — it raises AttributeError the moment adaptive thinking kicks in.
    Scan for the first block that actually has text instead.
    """
    for block in response.content:
        if hasattr(block, "text"):
            return block.text.strip()
    return ""


# ══════════════════════════════════════════════════════════════
# KNOWLEDGE BASE  (built in Lab C1)
# ══════════════════════════════════════════════════════════════

def _load_kb():
    kb_client = chromadb.Client()
    try:
        return kb_client.get_collection(KB_COLLECTION)
    except Exception:
        print("  KB collection not found — building it now from data/kb/ ...")
        from data.kb_setup import chunk_article  # reuse Lab C1's chunker

        collection = kb_client.create_collection(KB_COLLECTION)
        ids, docs, metas = [], [], []
        for md in sorted((Path(__file__).parent.parent / "data" / "kb").glob("*.md")):
            for c in chunk_article(md.read_text(encoding="utf-8"), md.name):
                ids.append(c["id"])
                docs.append(c["document"])
                metas.append(c["metadata"])
        collection.add(ids=ids, documents=docs, metadatas=metas)
        print(f"  Built KB with {len(ids)} chunks.")
        return collection


def _get_kb():
    global KB
    if KB is None:
        KB = _load_kb()
    return KB


# ══════════════════════════════════════════════════════════════
# TOOLS
# ══════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "get_contract",
        "description": (
            "Fetch one contract by its ID, including derived fields: approval_band, "
            "notice_state, utilisation_pct, days_to_renewal, proposed_uplift_pct."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "contract_id": {"type": "string", "description": "e.g. CTR-1004"}
            },
            "required": ["contract_id"],
        },
    },
    {
        "name": "search_policy",
        "description": (
            "Search the BizOps procurement policy knowledge base. Use this to find "
            "the rule that applies before making a recommendation. Call this at "
            "most twice per contract."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Plain-English policy question, e.g. 'who approves a 60 lakh contract'",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "find_category_overlap",
        "description": (
            "List every other active contract in the same category, with vendor, "
            "value and utilisation. Use this to judge whether consolidation is viable."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "e.g. Observability"}
            },
            "required": ["category"],
        },
    },
    {
        "name": "submit_recommendation",
        "description": "Record the final recommendation for this contract. Call exactly once, last.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contract_id": {"type": "string"},
                "recommendation": {
                    "type": "string",
                    "enum": ["RENEW", "RENEGOTIATE", "CONSOLIDATE", "TERMINATE"],
                },
                "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                "rationale": {
                    "type": "string",
                    "description": "Two or three sentences. State the specific numbers that drove the decision.",
                },
                "policy_citation": {
                    "type": "string",
                    "description": "The policy file and section that supports this, e.g. 'auto_renewal_rules.md §Escalation trigger'",
                },
                "estimated_annual_impact_inr": {
                    "type": "integer",
                    "description": "Rough rupee impact. 0 for RENEW at existing terms. Negative means saving.",
                },
                "human_approval_required": {
                    "type": "boolean",
                    "description": "True if policy requires a named human to approve before acting.",
                },
            },
            "required": [
                "contract_id",
                "recommendation",
                "confidence",
                "rationale",
                "policy_citation",
                "human_approval_required",
            ],
        },
    },
]


def tool_get_contract(contract_id: str) -> dict:
    try:
        r = requests.get(f"{CONTRACT_API}/api/contracts/{contract_id}", timeout=10)
        if r.status_code == 404:
            return {"error": f"{contract_id} not found"}
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Contract API unreachable ({e}). Is contract_shim.py running on port 5001?"}
    except ValueError:
        return {"error": f"Contract API returned non-JSON for {contract_id} (HTTP {r.status_code})."}


def tool_search_policy(query: str) -> dict:
    """Return whole policy sections, grouped by source article.

    Returning arbitrary top-N chunks scattered across files gives the model
    fragments with no context. Grouping by source and returning the best section
    per article keeps each result readable and citable.
    """
    try:
        res = _get_kb().query(query_texts=[query], n_results=6)
    except Exception as e:
        return {"error": f"Policy KB unavailable ({type(e).__name__}: {e}).", "results": []}

    best_per_source: dict[str, tuple[float, str, str]] = {}
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        src = meta.get("source", "unknown")
        heading = meta.get("heading", "(no heading)")
        if src not in best_per_source or dist < best_per_source[src][0]:
            best_per_source[src] = (dist, heading, doc)

    ranked = sorted(best_per_source.items(), key=lambda kv: kv[1][0])[:2]
    return {
        "results": [
            {
                "source": src,
                "section": heading,
                "confidence": round(1 - dist, 2),
                "text": doc,
            }
            for src, (dist, heading, doc) in ranked
        ]
    }


def tool_find_category_overlap(category: str) -> dict:
    try:
        r = requests.get(f"{CONTRACT_API}/api/categories", timeout=10)
        for c in r.json()["categories"]:
            if c["category"].lower() == category.lower():
                return c
        return {"category": category, "vendor_count": 0, "vendors": []}
    except requests.exceptions.RequestException as e:
        return {"error": f"Contract API unreachable ({e})."}
    except (ValueError, KeyError, TypeError) as e:
        return {"error": f"Unexpected /api/categories response ({e})."}


SYSTEM_PROMPT = """You are the Renewal Analysis Agent for Zensar BizOps.

For the contract you are given, decide one of: RENEW, RENEGOTIATE, CONSOLIDATE, TERMINATE.

Method — follow in order:
1. Call get_contract to read the real numbers. Never assume them.
2. Call search_policy to find the governing rule. Call it at most twice.
3. Call find_category_overlap ONLY if you are considering CONSOLIDATE.
4. Call submit_recommendation exactly once to finish.

Decision guidance:
- Utilisation below 40% with a viable overlapping vendor points to CONSOLIDATE.
- Utilisation below 20% with no business owner is not an automatic TERMINATE — an
  absent owner means escalate to a human, because nobody has confirmed the
  capability is unneeded.
- Proposed uplift above 15% is never accepted at first offer; that is RENEGOTIATE.
- High utilisation with modest uplift is a healthy RENEW.
- TERMINATE only where the capability itself is no longer required.

Confidence:
- HIGH means the numbers and the policy point the same way with no ambiguity.
- MEDIUM means the recommendation is sound but rests on an assumption you should name.
- LOW means genuinely unclear. LOW is a perfectly valid answer — say so rather than
  inventing certainty. Do not keep calling tools hoping for a cleaner picture.

Set human_approval_required to true whenever policy demands it (approval Bands B and C,
anything inside its notice window, and every termination).

Always cite the specific policy file and section that justifies your call."""


def analyse_contract(contract_id: str) -> dict | None:
    print(f"\n{'═' * 62}")
    print(f"ANALYSING: {contract_id}")
    print("═" * 62)

    messages = [
        {"role": "user", "content": f"Analyse contract {contract_id} and recommend an action."}
    ]
    rounds = 0

    while True:
        rounds += 1
        if rounds > MAX_ROUNDS:
            print(f"  ⚠️  Stopped after {MAX_ROUNDS} rounds with no recommendation — treating as LOW confidence.")
            return None

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            output_config={"effort": "medium"},
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            text = extract_text(response)
            if text:
                print(f"  Agent said: {text[:200]}")
            if response.stop_reason == "max_tokens":
                print("  ⚠️  Response hit max_tokens before calling a tool.")
            return None

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []

        for block in response.content:
            if block.type != "tool_use":
                continue

            name, args = block.name, block.input

            if name == "get_contract":
                result = tool_get_contract(args["contract_id"])
                if "error" in result:
                    print(f"  ✗ {result['error']}")
                else:
                    value = result.get("annual_value_inr")
                    value_s = f"{value:,}" if isinstance(value, (int, float)) else "?"
                    print(
                        f"  → contract: band {result.get('approval_band', '?')}  "
                        f"{result.get('notice_state', '?')}  util {result.get('utilisation_pct', '?')}%  "
                        f"uplift {result.get('proposed_uplift_pct', '?')}%  "
                        f"INR {value_s}"
                    )

            elif name == "search_policy":
                result = tool_search_policy(args["query"])
                if "error" in result:
                    print(f"  ✗ {result['error']}")
                tops = ", ".join(
                    f"{r['source']} §{r['section']} ({r['confidence']:.0%})"
                    for r in result["results"]
                )
                print(f'  → policy "{args["query"][:44]}" -> {tops or "no results"}')

            elif name == "find_category_overlap":
                result = tool_find_category_overlap(args["category"])
                if "error" in result:
                    print(f"  ✗ {result['error']}")
                names = ", ".join(
                    f"{v.get('vendor', '?')} ({v.get('utilisation_pct', '?')}%)"
                    for v in result.get("vendors", [])
                )
                print(f"  → overlap in {args['category']}: {names or 'none'}")

            elif name == "submit_recommendation":
                rec = args
                print(f"\n  ┌─ RECOMMENDATION ─────────────────────────────────")
                print(f"  │ {rec['recommendation']}   confidence {rec['confidence']}")
                print(f"  │ Policy: {rec['policy_citation']}")
                impact = rec.get("estimated_annual_impact_inr")
                if impact:
                    print(f"  │ Impact: INR {impact:,}/yr")
                print(f"  │ Human approval required: {rec['human_approval_required']}")
                print(f"  └──────────────────────────────────────────────────")
                print(f"  {rec['rationale']}")
                return rec

            else:
                result = {"error": f"unknown tool {name}"}

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                }
            )

        messages.append({"role": "user", "content": tool_results})


def main() -> None:
    global client

    api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY not set. Copy .env.template to .env and add your key,\n"
            'or run:  $env:ANTHROPIC_API_KEY = "sk-ant-..."'
        )
    client = anthropic.Anthropic(api_key=api_key)

    # A deliberate spread: healthy renewal, high uplift, low utilisation with an
    # overlap, an orphaned contract, and a high-value one inside its notice window.
    test_contracts = ["CTR-1003", "CTR-1012", "CTR-1005", "CTR-1006", "CTR-1004"]

    results = []
    for cid in test_contracts:
        try:
            rec = analyse_contract(cid)
        except anthropic.APIError as e:
            print(f"  ✗ API error: {e}")
            rec = None
        # Contracts with no decision still appear in the summary, flagged for a human.
        results.append(rec or {
            "contract_id": cid,
            "recommendation": "NO DECISION",
            "confidence": "LOW",
            "human_approval_required": True,
        })

    print(f"\n\n{'═' * 62}")
    print("PORTFOLIO SUMMARY")
    print("═" * 62)
    print(f"{'Contract':<11}{'Action':<14}{'Conf':<8}{'Approval?':<11}Impact")
    print("-" * 62)
    for r in results:
        impact = r.get("estimated_annual_impact_inr") or 0
        impact_s = f"INR {impact:,}" if impact else "—"
        print(
            f"{r['contract_id']:<11}{r['recommendation']:<14}{r['confidence']:<8}"
            f"{str(r['human_approval_required']):<11}{impact_s}"
        )
    needs_human = sum(1 for r in results if r["human_approval_required"])
    print("-" * 62)
    print(f"{len(results)} analysed · {needs_human} require human approval before action")


if __name__ == "__main__":
    main()