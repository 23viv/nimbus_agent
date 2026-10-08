"""
Migrate auth_users.json to MongoDB Atlas 'auth_users' collection.
Run this script ONCE to populate the collection.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from motor.motor_asyncio import AsyncIOMotorClient
import certifi

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "nimbus_db")

async def migrate():
    if not MONGODB_URI:
        print("MONGODB_URI not set. Please set it in .env.")
        return

    auth_file = Path(__file__).parent.parent / "data" / "auth_users.json"
    if not auth_file.exists():
        print(f"Auth file not found at {auth_file}")
        return

    with open(auth_file, "r", encoding="utf-8") as f:
        users = json.load(f)

    if not users:
        print("No users found in auth_users.json.")
        return

    print("Connecting to MongoDB Atlas...")
    client = AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
    db = client[MONGODB_DB_NAME]
    collection = db["auth_users"]

    # Clear existing users if you want a clean slate (optional)
    await collection.delete_many({})
    print("Cleared existing 'auth_users' collection.")

    # Insert users
    await collection.insert_many(users)
    print(f"Successfully inserted {len(users)} users into MongoDB.")

if __name__ == "__main__":
    asyncio.run(migrate())
