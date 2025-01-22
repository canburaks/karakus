import pytest
from typing import AsyncGenerator
from api.core.config import get_app_settings
from api.services.supabase.supabase_storage_service import SupabaseStorage
from api.services.supabase.config import get_supabase_settings

@pytest.fixture
def app_settings():
    return get_app_settings()

@pytest.fixture
async def storage_service() -> AsyncGenerator[SupabaseStorage, None]:
    """Create a test storage service instance"""
    yield SupabaseStorage()

@pytest.fixture(autouse=True)
async def cleanup_storage(storage_service: SupabaseStorage) -> AsyncGenerator[None, None]:
    """Cleanup any test buckets after each test"""
    yield
    try:
        buckets = await storage_service.list_buckets()
        for bucket in buckets:
            if bucket["name"].startswith("test-"):
                await storage_service.delete_bucket(bucket["name"])
    except Exception as e:
        print(f"Cleanup error: {str(e)}") 
