"""
Nimbus Support Agent — MongoDB Atlas Database Manager
Handles async message storage and session tracking via MongoDB Atlas (motor).
Falls back gracefully to a persistent local JSON file if MongoDB is unconfigured or unreachable.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "nimbus_db")

_client = None
_db = None
_collection = None
_mongo_ready = False

# Fallback session store: {session_id: [messages]}
_in_memory_sessions: dict[str, list[dict]] = {}

_LOCAL_FALLBACK_FILE = Path(__file__).parent.parent / "data" / "local_sessions.json"


def _load_local_fallback():
    global _in_memory_sessions
    if _LOCAL_FALLBACK_FILE.exists():
        try:
            with open(_LOCAL_FALLBACK_FILE, "r", encoding="utf-8") as f:
                _in_memory_sessions = json.load(f)
            print(f"[db] Loaded {len(_in_memory_sessions)} sessions from local JSON backup.")
        except Exception as e:
            print(f"[db] Failed to load local backup ({e}).")
            _in_memory_sessions = {}
    else:
        _in_memory_sessions = {}


def _save_local_fallback():
    try:
        _LOCAL_FALLBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_LOCAL_FALLBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(_in_memory_sessions, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[db] Failed to save local backup ({e}).")


async def init_db():
    """Initialize AsyncIOMotorClient connection to MongoDB Atlas."""
    global _client, _db, _collection, _mongo_ready

    # Always load local fallback so we have access to past sessions immediately
    _load_local_fallback()

    if not MONGODB_URI:
        print("[db] MONGODB_URI not set. Using local JSON fallback store.")
        _mongo_ready = False
        return

    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        import certifi
        _client = AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
        _db = _client[MONGODB_DB_NAME]
        _collection = _db["chat_messages"]
        # Ping database to verify connection
        await _client.admin.command('ping')
        _mongo_ready = True
        print(f"[db] MongoDB Atlas connected successfully (database: '{MONGODB_DB_NAME}').")
    except Exception as exc:
        print(f"[db] MongoDB Atlas connection failed ({exc}). Using local JSON fallback.")
        _mongo_ready = False


async def save_message(session_id: str, role: str, content: str, user_email: str = "") -> dict:
    """Save a chat message under a session_id, tagged with the user's email."""
    doc = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "user_email": user_email,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if _mongo_ready and _collection is not None:
        try:
            await _collection.insert_one(doc)
            # Remove MongoDB '_id' before returning
            doc.pop("_id", None)
            return doc
        except Exception as exc:
            print(f"[db] Error saving message to MongoDB ({exc}): falling back to memory.")

    # Fallback to in-memory store
    if session_id not in _in_memory_sessions:
        _in_memory_sessions[session_id] = []
    _in_memory_sessions[session_id].append(doc)
    _save_local_fallback()
    return doc


async def get_session_messages(session_id: str) -> list[dict]:
    """Retrieve all messages for a given session_id ordered by timestamp."""
    if _mongo_ready and _collection is not None:
        try:
            cursor = _collection.find({"session_id": session_id}).sort("timestamp", 1)
            messages = []
            async for doc in cursor:
                doc.pop("_id", None)
                messages.append(doc)
            return messages
        except Exception as exc:
            print(f"[db] Error retrieving messages from MongoDB ({exc}).")

    # Fallback to in-memory store
    return _in_memory_sessions.get(session_id, [])


async def get_all_sessions(user_email: str = "") -> list[dict]:
    """
    Retrieve list of unique sessions with their message count and last updated timestamp.
    If user_email is provided, only return sessions belonging to that user.
    """
    if _mongo_ready and _collection is not None:
        try:
            match_stage = {}
            if user_email:
                match_stage = {"$match": {"user_email": user_email}}

            pipeline = []
            if match_stage:
                pipeline.append(match_stage)
            pipeline.extend([
                {
                    "$group": {
                        "_id": "$session_id",
                        "message_count": {"$sum": 1},
                        "last_updated": {"$max": "$timestamp"},
                        "first_message": {"$first": "$content"},
                        "user_email": {"$first": "$user_email"},
                    }
                },
                {"$sort": {"last_updated": -1}},
            ])
            cursor = _collection.aggregate(pipeline)
            sessions = []
            async for doc in cursor:
                sessions.append({
                    "session_id": doc["_id"],
                    "message_count": doc["message_count"],
                    "last_updated": doc["last_updated"],
                    "preview": doc.get("first_message", "")[:50],
                    "user_email": doc.get("user_email", ""),
                })
            return sessions
        except Exception as exc:
            print(f"[db] Error retrieving session list from MongoDB ({exc}).")

    # Fallback to in-memory store
    sessions = []
    for sid, msgs in _in_memory_sessions.items():
        if msgs:
            # Filter by user_email if provided
            session_email = msgs[0].get("user_email", "")
            if user_email and session_email != user_email:
                continue
            sessions.append({
                "session_id": sid,
                "message_count": len(msgs),
                "last_updated": msgs[-1]["timestamp"],
                "preview": msgs[0]["content"][:50],
                "user_email": session_email,
            })
    sessions.sort(key=lambda s: s["last_updated"], reverse=True)
    return sessions


async def clear_session(session_id: str) -> bool:
    """Clear all messages for a specific session."""
    if _mongo_ready and _collection is not None:
        try:
            await _collection.delete_many({"session_id": session_id})
        except Exception as exc:
            print(f"[db] Error clearing session in MongoDB ({exc}).")

    if session_id in _in_memory_sessions:
        del _in_memory_sessions[session_id]
        _save_local_fallback()

    return True
