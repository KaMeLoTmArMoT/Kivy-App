import os

from app.screens.utils.custom_logging import get_logger
from app.screens.utils.db import DB
from app.screens.utils.utils import extend_key, get_sha

logger = get_logger(__name__)


class AuthService:
    """Domain service managing login authentication, password registration, and encryption keys."""

    def __init__(self, db: DB = None):
        self.db = db or DB()

    def get_source_name(self) -> str:
        """Return database password destination key depending on environment."""
        return "login_test" if os.environ.get("APP_ENV") == "test" else "login"

    def get_stored_passwords(self, source_name: str) -> list:
        """Fetch password records for destination."""
        return self.db.get_login_password(source_name)

    def is_registered(self, source_name: str) -> bool:
        """Check if a password has been set."""
        return bool(self.get_stored_passwords(source_name))

    def validate_password(self, input_password: str, source_name: str) -> tuple[bool, str]:
        """Validate input password against stored hash."""
        records = self.get_stored_passwords(source_name)
        if not records:
            return False, "No password registered."

        real_hash, input_hash = records[0][1], get_sha(input_password)
        logger.debug(f"Validating password: input hash {input_hash}, real hash {real_hash}")
        if real_hash == input_hash:
            return True, "Valid password."
        return False, "Wrong password. Try again."

    def register_password(self, new_password: str, source_name: str) -> str:
        """Register a new login password in database."""
        enc_pass = get_sha(new_password)
        logger.debug(f"Registering new password hash {enc_pass} for {source_name}")
        self.db.set_login_password(enc_pass, origin=source_name)
        return enc_pass

    def derive_key(self, password: str) -> bytes:
        """Derive AES key from password string."""
        return extend_key(password)
