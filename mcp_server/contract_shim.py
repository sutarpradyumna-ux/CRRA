"""CRRA Lab C2 - Mock Contract API. Run from project root: python mcp_server/contract_shim.py

Sample checks:  /api/contracts/expiring?days=60   /api/contracts?band=C   /api/categories
"""

import csv
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, jsonify, request

app = Flask(__name__)

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "contracts.csv"
SIMULATED_TODAY = datetime(2025, 4, 1)
INT_FIELDS = ("annual_value_inr", "notice_days", "seats_purchased",
              "seats_active", "proposed_uplift_pct")
PATCHABLE = ("status", "owner", "proposed_uplift_pct")
CONTRACTS: list[dict] = []


def enrich(row: dict) -> dict:
    """Convert types and add the four derived fields."""
    for f in INT_FIELDS:
        row[f] = int(row[f])
    row["auto_renew"] = row["auto_renew"].strip().upper() == "Y"

    renewal = datetime.strptime(row["renewal_date"], "%Y-%m-%d")
    deadline = renewal - timedelta(days=row["notice_days"])
    row["notice_deadline"] = deadline.strftime("%Y-%m-%d")
    row["days_to_renewal"] = (renewal - SIMULATED_TODAY).days
    row["days_to_notice_deadline"] = (deadline - SIMULATED_TODAY).days

    if renewal < SIMULATED_TODAY:
        row["notice_state"] = "EXPIRED"
    elif deadline <= SIMULATED_TODAY:
        row["notice_state"] = "INSIDE_WINDOW"
    elif row["days_to_notice_deadline"] <= 30:
        row["notice_state"] = "APPROACHING"
    else:
        row["notice_state"] = "OPEN"

    seats = row["seats_purchased"]
    row["utilisation_pct"] = round(100 * row["seats_active"] / seats) if seats else None

    v = row["annual_value_inr"]
    row["approval_band"] = "A" if v < 1_000_000 else ("B" if v <= 5_000_000 else "C")
    return row


def load_contracts() -> None:
    CONTRACTS.clear()
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        CONTRACTS.extend(enrich(row) for row in csv.DictReader(f))


def find(contract_id: str):
    return next((c for c in CONTRACTS if c["contract_id"].upper() == contract_id.upper()), None)


def summarise(items: list[dict]) -> dict:
    """Totals and breakdowns for any result set, so callers need no arithmetic."""
    by_band, by_state = {}, {}
    for c in items:
        by_band[c["approval_band"]] = by_band.get(c["approval_band"], 0) + 1
        by_state[c["notice_state"]] = by_state.get(c["notice_state"], 0) + 1
    return {
        "total_annual_value_inr": sum(c["annual_value_inr"] for c in items),
        "by_band": dict(sorted(by_band.items())),
        "by_notice_state": dict(sorted(by_state.items())),
    }


def not_found(contract_id: str):
    return jsonify({"error": f"Contract {contract_id} not found"}), 404


@app.get("/health")
def health():
    return jsonify({"status": "ok", "contracts_loaded": len(CONTRACTS),
                    "simulated_today": SIMULATED_TODAY.strftime("%Y-%m-%d")})


@app.get("/api/contracts")
def list_contracts():
    results = CONTRACTS
    for param, field in (("category", "category"), ("band", "approval_band"),
                         ("notice_state", "notice_state")):
        value = request.args.get(param)
        if value:
            results = [c for c in results if c[field].lower() == value.lower()]
    return jsonify({"count": len(results), "summary": summarise(results), "contracts": results})


@app.get("/api/contracts/expiring")
def expiring():
    try:
        days = int(request.args.get("days", 90))
    except ValueError:
        return jsonify({"error": "days must be an integer"}), 400
    results = sorted((c for c in CONTRACTS if 0 <= c["days_to_renewal"] <= days),
                     key=lambda c: c["days_to_renewal"])
    return jsonify({"count": len(results), "window_days": days,
                    "summary": summarise(results), "contracts": results})


@app.get("/api/contracts/<contract_id>")
def get_contract(contract_id):
    c = find(contract_id)
    return jsonify(c) if c else not_found(contract_id)


@app.get("/api/categories")
def categories():
    grouped: dict[str, list[dict]] = {}
    for c in CONTRACTS:
        grouped.setdefault(c["category"], []).append(
            {k: c[k] for k in ("contract_id", "vendor", "annual_value_inr", "utilisation_pct")})
    summary = [{"category": cat, "vendor_count": len(items),
                "total_annual_value_inr": sum(i["annual_value_inr"] for i in items),
                "vendors": items} for cat, items in grouped.items()]
    summary.sort(key=lambda s: (-s["vendor_count"], -s["total_annual_value_inr"]))  # most vendors first
    top = summary[0]["vendor_count"] if summary else 0
    most = [{k: s[k] for k in ("category", "vendor_count", "total_annual_value_inr")}
            for s in summary if s["vendor_count"] == top]
    return jsonify({"count": len(summary), "most_vendors": most, "categories": summary})


@app.patch("/api/contracts/<contract_id>")
def update_contract(contract_id):
    """In-memory only - restarting the server resets every change."""
    c = find(contract_id)
    if not c:
        return not_found(contract_id)
    payload = request.get_json(silent=True) or {}
    changes = {k: payload[k] for k in PATCHABLE if k in payload}
    if not changes:
        return jsonify({"error": f"Send at least one of: {', '.join(PATCHABLE)}"}), 400
    c.update(changes)
    return jsonify({"updated": True, "changed": list(changes), "contract": c})


load_contracts()

if __name__ == "__main__":
    print(f"Mock Contract API | {len(CONTRACTS)} contracts | "
          f"simulated today {SIMULATED_TODAY:%Y-%m-%d} | http://localhost:5001/health")
    app.run(port=5001, debug=False)