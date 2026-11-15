import os
from pathlib import Path
import sys
import tempfile

import pytest

TEST_ROOT = Path(tempfile.mkdtemp(prefix="document-verification-tests-"))
os.environ["DATABASE_URL"] = f"sqlite:///{(TEST_ROOT / 'test.db').as_posix()}"
os.environ["UPLOAD_DIR"] = str(TEST_ROOT / "uploads")
os.environ["JWT_SECRET"] = "test-only-signing-secret-that-is-long-enough-1234567890"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import Base, engine, SessionLocal  # noqa: E402
from app.main import app  # noqa: E402
from app.models import AdminUser  # noqa: E402
from app.security import hash_password  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as test_client:
        yield test_client
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def signed_in_client(client):
    with SessionLocal() as db:
        db.add(
            AdminUser(
                username="test-admin",
                password_hash=hash_password("test-password-with-sufficient-length"),
                role="admin",
            )
        )
        db.commit()
    response = client.post(
        "/auth/login",
        json={"username": "test-admin", "password": "test-password-with-sufficient-length"},
    )
    assert response.status_code == 200
    client.headers.update({"Authorization": f"Bearer {response.json()['access_token']}"})
    return client
