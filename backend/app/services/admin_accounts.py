from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import AdminUser
from app.security import hash_password


def register_admin(db: Session, username: str, password: str, role: str = "admin") -> AdminUser:
    username = username.strip()
    if not 3 <= len(username) <= 64:
        raise ValueError("Username must be between 3 and 64 characters.")
    if role not in {"admin", "reviewer"}:
        raise ValueError("Role must be admin or reviewer.")
    if len(password) < 12:
        raise ValueError("Password must be at least 12 characters.")
    if db.scalar(select(AdminUser.id).where(AdminUser.username == username)):
        raise ValueError("That administrator username already exists.")
    admin = AdminUser(username=username, password_hash=hash_password(password), role=role)
    db.add(admin)
    try:
        db.commit()
        db.refresh(admin)
    except IntegrityError as exc:
        db.rollback()
        raise ValueError("That administrator username already exists.") from exc
    return admin
