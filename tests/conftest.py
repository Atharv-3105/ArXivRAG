import pytest 
from fastapi.testclient import TestClient 
from src.api.app import create_app

@pytest.fixture(scope = "module")
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client