"""
Nimbus Support Agent — MCP Server
Exposes user lookup tools via FastMCP over stdio transport.
"""

import json
import os
from pathlib import Path
from fastmcp import FastMCP

# ── Load user database ────────────────────────────────────────────────────────
_DB_PATH = Path(__file__).parent.parent / "data" / "users.json"

def _load_users() -> list[dict]:
    with open(_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

USERS: list[dict] = _load_users()

# ── FastMCP server ─────────────────────────────────────────────────────────────
mcp = FastMCP("nimbus-user-db")


@mcp.tool
def get_user_by_email(email: str) -> dict:
    """
    Look up a Nimbus customer account by email address.
    Returns the user record including name, plan, account_status, last_login,
    created_at, total_orders, and store_credit.
    Returns an error dict if no user is found.
    """
    email = email.strip().lower()
    for user in USERS:
        if user["email"].lower() == email:
            # Return a clean, safe subset of the record
            return {
                "found": True,
                "id": user["id"],
                "name": user["name"],
                "email": user["email"],
                "plan": user["plan"],
                "account_status": user["account_status"],
                "last_login": user["last_login"],
                "created_at": user["created_at"],
                "total_orders": user["total_orders"],
                "store_credit": user["store_credit"],
            }
    return {
        "found": False,
        "error": f"No Nimbus account found for email: {email}",
    }


@mcp.tool
def get_user_account_status(user_id: str) -> dict:
    """
    Retrieve the account plan and status details for a Nimbus user by their user ID.
    Returns plan type, account status, and (if suspended) the suspension reason.
    Returns an error dict if the user_id is not found.
    """
    user_id = user_id.strip()
    for user in USERS:
        if user["id"] == user_id:
            result = {
                "found": True,
                "id": user["id"],
                "name": user["name"],
                "plan": user["plan"],
                "account_status": user["account_status"],
            }
            if user["account_status"] == "suspended" and "suspension_reason" in user:
                result["suspension_reason"] = user["suspension_reason"]
            # Plan details
            if user["plan"] == "premium":
                result["plan_details"] = (
                    "Nimbus Premium: free standard shipping, 10% member discount, "
                    "early access to sales, priority support. $9.99/month."
                )
            else:
                result["plan_details"] = (
                    "Nimbus Free: standard shipping rates apply, regular support. "
                    "Upgrade to Premium for $9.99/month."
                )
            return result
    return {
        "found": False,
        "error": f"No Nimbus account found for user_id: {user_id}",
    }


if __name__ == "__main__":
    mcp.run()
