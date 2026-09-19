import uuid
from datetime import datetime


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ArXivMind"


def test_versioned_health_endpoint(client):
    assert client.get("/api/v1/health").status_code == 200


def test_health_timestamp_is_valid_iso8601(client):
    timestamp = client.get("/health").json()["timestamp"]
    parsed = datetime.fromisoformat(timestamp)
    assert parsed.tzinfo is not None


def test_trace_id_is_generated_when_absent(client):
    trace_id = client.get("/health").headers["X-Trace-ID"]
    uuid.UUID(trace_id)  # raises if not a UUID


def test_trace_id_from_caller_is_propagated(client):
    response = client.get("/health", headers={"X-Trace-ID": "caller-supplied-123"})
    assert response.headers["X-Trace-ID"] == "caller-supplied-123"
