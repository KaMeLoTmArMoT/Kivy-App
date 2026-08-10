from base64 import b64decode, b64encode

from Cryptodome.Cipher import AES

from app.screens.utils.custom_logging import get_logger
from app.screens.utils.db import DB

logger = get_logger(__name__)


class CustomerService:
    """Domain service managing AES-encrypted customer records in sqlite database."""

    def __init__(self, db: DB = None):
        self.db = db or DB()

    def encrypt_text(self, text: str, key: bytes) -> str:
        """Encrypt text string using AES MODE_EAX."""
        cipher = AES.new(key, AES.MODE_EAX, nonce=b"TODO")
        encoded_bytes = cipher.encrypt(text.encode("utf-8"))
        return b64encode(encoded_bytes).decode("utf-8")

    def decrypt_text(self, encrypted_text: str, key: bytes) -> str:
        """Decrypt AES MODE_EAX encoded string."""
        cipher = AES.new(key, AES.MODE_EAX, nonce=b"TODO")
        decoded_bytes = b64decode(encrypted_text.encode("utf-8"))
        return cipher.decrypt(decoded_bytes).decode("utf-8")

    def add_customer(self, name: str, key: bytes) -> None:
        """Encrypt and insert customer name record."""
        b_encoded = self.encrypt_text(name, key)
        self.db.insert_customer(b_encoded)
        logger.info(f"Added customer record: {name}")

    def get_decrypted_customers(self, key: bytes) -> list[tuple[str, str]]:
        """Fetch all customer records and return tuples of (decrypted_name, raw_encrypted)."""
        raw_records = self.db.get_customers()
        results = []
        for row in raw_records:
            raw_enc = row[0]
            decrypted_name = self.decrypt_text(raw_enc, key)
            results.append((decrypted_name, raw_enc))
        return results

    def delete_customer_by_name(self, name: str, key: bytes) -> None:
        """Delete customer record matching plaintext name."""
        b_encoded = self.encrypt_text(name, key)
        self.db.delete_customer(b_encoded)
        logger.info(f"Deleted customer record: {name}")

    def update_customer_name(self, old_name: str, new_name: str, key: bytes) -> None:
        """Update customer record from old_name to new_name."""
        old_enc = self.encrypt_text(old_name, key)
        new_enc = self.encrypt_text(new_name, key)
        self.db.update_customer(new_enc, old_enc)
        logger.info(f"Updated customer record from {old_name} to {new_name}")
