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
        cipher = AES.new(key, AES.MODE_EAX)
        ciphertext, tag = cipher.encrypt_and_digest(text.encode("utf-8"))
        payload = b"v2" + cipher.nonce + tag + ciphertext
        return b64encode(payload).decode("utf-8")

    def decrypt_text(self, encrypted_text: str, key: bytes) -> str:
        """Decrypt AES MODE_EAX encoded string."""
        payload = b64decode(encrypted_text.encode("utf-8"))
        if payload.startswith(b"v2"):
            nonce, tag, ciphertext = payload[2:18], payload[18:34], payload[34:]
            cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
            return cipher.decrypt_and_verify(ciphertext, tag).decode("utf-8")

        # Read records written by the pre-migration fixed-nonce format.
        cipher = AES.new(key, AES.MODE_EAX, nonce=b"TODO")
        return cipher.decrypt(payload).decode("utf-8")

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
        for decrypted_name, raw_encrypted in self.get_decrypted_customers(key):
            if decrypted_name == name:
                self.db.delete_customer(raw_encrypted)
                break
        logger.info(f"Deleted customer record: {name}")

    def update_customer_name(self, old_name: str, new_name: str, key: bytes) -> None:
        """Update customer record from old_name to new_name."""
        for decrypted_name, raw_encrypted in self.get_decrypted_customers(key):
            if decrypted_name == old_name:
                self.db.update_customer(self.encrypt_text(new_name, key), raw_encrypted)
                break
        logger.info(f"Updated customer record from {old_name} to {new_name}")
