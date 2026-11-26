import os
import pytest
import requests

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def test_recommend_valid_payload():
    payload = {"user_id": "00000dbacae5abe5e23885899a1fa44253a17956c6d1c3d25f88aa139fdf6654", "k": 10}
    resp = requests.post(f"{BASE_URL}/recommend", json=payload, timeout=5)
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code} {resp.text}"
    data = resp.json()
    assert isinstance(data, dict)
    recs = data.get("recommendations") or data.get("items") or []
    assert isinstance(recs, list)
    assert 0 < len(recs) <= 10


def test_recommend_cold_user_fallback():
    payload = {"user_id": "cold_or_unknown_user_000", "k": 5}
    resp = requests.post(f"{BASE_URL}/recommend", json=payload, timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    recs = data.get("recommendations") or data.get("items") or []
    assert isinstance(recs, list)
    assert len(recs) == 5


@pytest.mark.parametrize("bad_k", [0, -1])
def test_recommend_invalid_k(bad_k):
    payload = {"user_id": "test_user", "k": bad_k}
    resp = requests.post(f"{BASE_URL}/recommend", json=payload, timeout=5)
    assert resp.status_code in (400, 422)
