import base64
import hashlib
import hmac
import secrets
import time

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError, VerificationError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.config import AUTH_TOKEN_TTL_SECONDS, JWT_SECRET
from app.database import get_db
from app.models import AdminUser

bearer = HTTPBearer(auto_error=False)
password_hasher = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2)
LEGACY_PBKDF2_ROUNDS = 310_000


def hash_password(password: str) -> str:
    return password_hasher.hash(password)


def verify_password(password: str, encoded: str) -> bool:
    if encoded.startswith("$argon2id$"):
        try:
            return password_hasher.verify(encoded, password)
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return False
    try:
        algorithm, rounds, salt_text, digest_text = encoded.split("$")
        if algorithm != "pbkdf2_sha256" or int(rounds) > 2_000_000:
            return False
        salt = base64.urlsafe_b64decode(salt_text)
        expected = base64.urlsafe_b64decode(digest_text)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(rounds))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def password_needs_upgrade(encoded: str) -> bool:
    if not encoded.startswith("$argon2id$"):
        return True
    try:
        return password_hasher.check_needs_rehash(encoded)
    except (VerificationError, InvalidHashError):
        return True


def issue_token(admin: AdminUser) -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "sub": str(admin.id),
            "username": admin.username,
            "role": admin.role,
            "iat": now,
            "exp": now + AUTH_TOKEN_TTL_SECONDS,
            "jti": secrets.token_urlsafe(16),
        },
        JWT_SECRET,
        algorithm="HS256",
    )


def decode_token(token: str) -> dict:
    try:
        claims = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=["HS256"],
            options={"require": ["sub", "exp", "iat", "jti"]},
        )
        return claims
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def require_admin(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> AdminUser:
    admin = _authenticated_user(credentials, db)
    if admin.role != "admin":
        raise HTTPException(status_code=403, detail="Administrator role required")
    return admin


def require_staff(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> AdminUser:
    admin = _authenticated_user(credentials, db)
    if admin.role not in {"admin", "reviewer"}:
        raise HTTPException(status_code=403, detail="Staff role required")
    return admin


def _authenticated_user(
    credentials: HTTPAuthorizationCredentials | None,
    db: Session,
) -> AdminUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Administrator login required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    claims = decode_token(credentials.credentials)
    try:
        admin_id = int(claims["sub"])
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=401, detail="Invalid token subject") from exc
    admin = db.get(AdminUser, admin_id)
    if admin is None or not admin.is_active:
        raise HTTPException(status_code=401, detail="Administrator account is unavailable")
    return admin
