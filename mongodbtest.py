import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

if MONGO_URI:
    print(f"DEBUG: MONGO_URI loaded successfully: {MONGO_URI[:30]}...") # Shows first 30 chars
else:
    print("DEBUG: MONGO_URI is NOT loaded. Check .env file path and content.")