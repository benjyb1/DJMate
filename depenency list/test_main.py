"""
Basic tests for DJMate API endpoints.

Run with: pytest
"""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root_endpoint():
    """Test the health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["version"] == "2.0.0"


def test_parse_intent():
    """Test natural language parsing endpoint."""
    payload = {
        "query": "Find me some deep house tracks around 120 BPM"
    }
    response = client.post("/parse-intent", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "original_query" in data
    assert "structured_query" in data
    assert "confidence" in data


def test_intelligent_recommend():
    """Test recommendation endpoint."""
    payload = {
        "tags": ["deep house"],
        "bpm_range": [118, 122]
    }
    response = client.post("/intelligent-recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert "reasoning" in data


def test_crate_operations():
    """Test crate management endpoint."""
    payload = {
        "session_id": "test-session-123",
        "tracks": ["track1", "track2"],
        "sequence_order": [0, 1]
    }
    response = client.post("/crate/operations", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "compatibility_issues" in data
    assert "sequence_score" in data


def test_pathway_visualization():
    """Test pathway visualization endpoint."""
    response = client.get(
        "/visualization/pathway",
        params={
            "from_track": "track1",
            "to_tracks": ["track2", "track3"]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
