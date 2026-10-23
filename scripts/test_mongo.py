import asyncio
import os
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from motor.motor_asyncio import AsyncIOMotorClient
import certifi

async def main():
    uri = os.getenv('MONGODB_URI')
    if not uri:
        print("NO MONGODB_URI FOUND")
        return
    client = AsyncIOMotorClient(uri, tlsCAFile=certifi.where())
    db = client[os.getenv('MONGODB_DB_NAME', 'nimbus_db')]
    coll = db['auth_users']
    try:
        users = await coll.find().to_list(length=100)
        print('Users in MongoDB:', [u.get('email') for u in users])
    except Exception as e:
        print('MONGO ERROR:', e)

if __name__ == '__main__':
    asyncio.run(main())
