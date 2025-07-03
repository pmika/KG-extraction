import requests
import json
from typing import Union, Optional

class JSONLDGraphDBStorage:
    def __init__(self, repo_id: str, base_url: str = "http://localhost:7200"):
        """
        Initialize the storage with the GraphDB repository ID and base URL.
        Args:
            repo_id (str): The repository ID in GraphDB.
            base_url (str): The base URL for the GraphDB instance (default: http://localhost:7200).
        """
        self.repo_id = repo_id
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/repositories/{self.repo_id}/statements"

    def upload_jsonld(self, jsonld_data: Union[dict, str], context: Optional[str] = None) -> bool:
        """
        Upload JSON-LD data to the GraphDB repository.
        Args:
            jsonld_data (Union[dict, str]): The JSON-LD data to upload.
            context (Optional[str]): Optional named graph context (as a full IRI, e.g. '<http://example.org/graph>').
        Returns:
            bool: True if upload succeeded, False otherwise.
        """
        headers = {"Content-Type": "application/ld+json"}
        data = json.dumps(jsonld_data) if isinstance(jsonld_data, dict) else jsonld_data
        url = self.endpoint
        if context:
            from urllib.parse import quote
            # GraphDB expects the context to be URL-encoded and wrapped in <>
            encoded_context = quote(context, safe='')
            url += f"?context={encoded_context}"
        try:
            response = requests.post(url, headers=headers, data=data)
            if response.status_code in (200, 204):
                print(f"Successfully uploaded JSON-LD to GraphDB repo '{self.repo_id}'.")
                return True
            else:
                print(f"Failed to upload JSON-LD: {response.status_code} {response.text}")
                return False
        except Exception as e:
            print(f"Error uploading JSON-LD to GraphDB: {e}")
            return False 