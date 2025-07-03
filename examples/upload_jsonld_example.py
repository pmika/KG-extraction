import json
from src.storage.jsonld_graphdb_storage import JSONLDGraphDBStorage

# Path to the JSON-LD file to upload
jsonld_file = "extracted_knowledge.jsonld"
# GraphDB repository ID (change as needed)
repo_id = "test"

# Read JSON-LD data from file
try:
    with open(jsonld_file, "r", encoding="utf-8") as f:
        jsonld_data = json.load(f)
except Exception as e:
    print(f"Failed to read JSON-LD file: {e}")
    exit(1)

# Initialize the storage uploader
storage = JSONLDGraphDBStorage(repo_id=repo_id)

# Upload the JSON-LD data to GraphDB
did_upload = storage.upload_jsonld(jsonld_data)

if did_upload:
    print("Upload successful!")
else:
    print("Upload failed.") 