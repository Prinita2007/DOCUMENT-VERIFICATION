import argparse
from getpass import getpass
import os
from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_backend_dir = _project_root / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from app.database import SessionLocal, validate_database
from app.services.admin_accounts import register_admin


def main() -> None:
    validate_database()
    parser = argparse.ArgumentParser(description="Create an administrator account.")
    parser.add_argument("--username", help="Administrator username")
    parser.add_argument("--password", help="Administrator password (min 12 chars)")
    parser.add_argument("--role", default="admin", choices=["admin", "reviewer"], help="User role (admin or reviewer)")
    args = parser.parse_args()

    username = args.username or os.getenv("INITIAL_ADMIN_USERNAME")
    role = args.role
    password = args.password or os.getenv("INITIAL_ADMIN_PASSWORD")

    if not username:
        username = input("New administrator username: ").strip()
    if not args.username and not args.role:
        role_input = input("Role (admin/reviewer, default admin): ").strip().lower()
        if role_input in ("admin", "reviewer"):
            role = role_input
    if not password:
        password = getpass("New administrator password (minimum 12 characters): ")
        confirmation = getpass("Confirm password: ")
        if password != confirmation:
            raise SystemExit("Passwords do not match.")

    with SessionLocal() as db:
        try:
            admin = register_admin(db, username, password, role)
            print(f"Administrator account '{admin.username}' ({admin.role}) created successfully.")
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
