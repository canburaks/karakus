import pytest
from api.services.supabase.supabase_storage_service import SupabaseStorage

@pytest.mark.asyncio
async def test_storage_flow(storage_service: SupabaseStorage):
    # Test bucket operations
    bucket_name = "test-bucket"
    await storage_service.create_bucket(bucket_name, is_public=True)
    
    buckets = await storage_service.list_buckets()
    assert any(b["name"] == bucket_name for b in buckets)
    
    # Test file operations
    test_content = b"test content"
    await storage_service.upload_file(
        bucket_name,
        "test.txt",
        test_content,
        content_type="text/plain"
    )
    
    files = await storage_service.list_files(bucket_name)
    assert any(f["name"] == "test.txt" for f in files)
    
    # Test signed URL
    signed_url = await storage_service.create_signed_url(
        bucket_name,
        "test.txt",
        expires_in=60
    )
    assert "signedURL" in signed_url
    
    # Test move operation
    await storage_service.move_file(
        bucket_name,
        "test.txt",
        "moved-test.txt"
    )
    
    files = await storage_service.list_files(bucket_name)
    assert any(f["name"] == "moved-test.txt" for f in files)
    
    content = await storage_service.download_file(bucket_name, "moved-test.txt")
    assert content == test_content
    
    # Cleanup
    await storage_service.delete_file(bucket_name, ["moved-test.txt"])
    await storage_service.delete_bucket(bucket_name) 
