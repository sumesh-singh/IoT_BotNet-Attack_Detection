import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from api.main import app


def test_health():
    client = TestClient(app)
    r = client.get('/api/v1/health')
    assert r.status_code == 200
    data = r.json()
    assert data.get('status') == 'ok'
