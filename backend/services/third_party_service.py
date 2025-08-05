# backend/services/third_party_service.py

import logging
from typing import List, Dict, Any
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from requests_oauthlib import OAuth2Session
from config.config_manager import config_manager

logger = logging.getLogger(__name__)

class ThirdPartyService:
    """
    A service for interacting with third-party data sources like Google Drive, OneDrive, etc.
    """
    def __init__(self, config_manager_instance):
        self.config_manager = config_manager_instance
        logger.info("ThirdPartyService initialized.")

    async def connect_google_drive(self, user_id: str, credentials_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Connects a user's Google Drive using their OAuth 2.0 credentials.
        """
        client_id = self.config_manager.get_secret("google_drive_client_id")
        client_secret = self.config_manager.get_secret("google_drive_client_secret")
        # In a real app, you would handle the full OAuth 2.0 flow here.

        return {"success": True, "message": "Successfully connected to Google Drive."}

    async def query_google_drive(self, user_id: str, query: str, credentials_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Queries a user's Google Drive."""
        creds = Credentials(**credentials_info)
        service = build('drive', 'v3', credentials=creds)

        results = service.files().list(
            q=f"name contains '{query}'",
            pageSize=10,
            fields="nextPageToken, files(id, name, mimeType)"
        ).execute()

        items = results.get('files', [])
        return [{"name": item['name'], "type": item['mimeType'], "source": "Google Drive"} for item in items]

    async def connect_one_drive(self, user_id: str, token: Dict[str, Any]) -> Dict[str, Any]:
        """Connects a user's OneDrive using their OAuth 2.0 token."""
        client_id = self.config_manager.get_secret("onedrive_client_id")
        client_secret = self.config_manager.get_secret("onedrive_client_secret")
        # In a real app, you would handle the full OAuth 2.0 flow here.

        return {"success": True, "message": "Successfully connected to OneDrive."}

    async def query_one_drive(self, user_id: str, query: str, token: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Queries a user's OneDrive."""
        graph_url = 'https://graph.microsoft.com/v1.0'

        onedrive = OAuth2Session(token=token)
        response = onedrive.get(f"{graph_url}/me/drive/root/search(q='{query}')")

        items = response.json().get('value', [])
        return [{"name": item['name'], "type": item.get('file', {}).get('mimeType'), "source": "OneDrive"} for item in items]

third_party_service = ThirdPartyService(config_manager)
