# backend/services/storage_service.py

import logging
from shared_tools.cloud_storage_utils import CloudStorageUtilsWrapper
from config.config_manager import config_manager

logger = logging.getLogger(__name__)

class StorageService:
    """
    A service for managing user storage.
    """
    def __init__(self, cloud_storage_utils: CloudStorageUtilsWrapper):
        self.cloud_storage_utils = cloud_storage_utils
        logger.info("StorageService initialized.")

    async def get_user_storage_usage(self, user_id: str) -> float:
        """
        Calculates the storage usage for a given user in MB.
        """
        result = await self.cloud_storage_utils.list_user_files(user_id)
        if result["success"]:
            return result["total_size_mb"]
        else:
            # Handle the case where the storage usage could not be retrieved
            logger.error(f"Could not retrieve storage usage for user {user_id}: {result.get('message')}")
            return 0.0

storage_service = StorageService(cloud_storage_utils=CloudStorageUtilsWrapper(config_manager))
