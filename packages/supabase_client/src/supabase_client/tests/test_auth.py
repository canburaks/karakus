from fastapi.testclient import TestClient

from ..main import app


@app.get("/")
async def read_main():
    return {"msg": "Hello World"}


client = TestClient(app=app)


def test_anonymous_access():
    response = client.get("/auth")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}
