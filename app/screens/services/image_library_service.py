import hashlib
from pathlib import Path
from shutil import copy

from checksumdir import dirhash
from Cryptodome.Cipher import AES

from app.screens.utils.custom_logging import get_logger
from app.screens.utils.db import DB

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def guess_image_ext(data: bytes) -> str | None:
    """Determine image extension from binary header signature."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if data.startswith(b"BM"):
        return "bmp"
    return None


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ImageLibraryService:
    """Domain service managing disk image browsing, AES DB image encryption, and transfers."""

    def __init__(self, db: DB = None):
        self.db = db or DB()

    def get_supported_images_in_dir(self, directory_path: str) -> list[str]:
        """Scan directory and return list of valid image file paths."""
        path = Path(directory_path)
        if not path.is_dir():
            return []
        return sorted(str(p) for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    @staticmethod
    def get_directory_hash(directory_path: str) -> str:
        path = Path(directory_path)
        return dirhash(str(path), "sha1") if path.is_dir() else ""

    def save_file_to_db(
        self, file_path: str, key: bytes | None = None, encrypt: bool = False
    ) -> None:
        """Read image from disk and insert into database (encrypted or plaintext)."""
        blob_data = Path(file_path).read_bytes()

        if encrypt and key:
            cipher = AES.new(key, AES.MODE_EAX)
            ciphertext, tag = cipher.encrypt_and_digest(blob_data)
            blob_data = cipher.nonce + tag + ciphertext

        self.db.insert_image(blob_data)

    def load_and_decode_db_images(
        self, key: bytes
    ) -> tuple[list[tuple[int, bytes, str]], list[tuple[int, bytes, str]]]:
        """
        Fetch all DB images and decode them into (plain_images, secure_images).
        Returns lists of tuples: (pk, decoded_bytes, extension)
        """
        db_records = self.db.get_images()
        plain_images = []
        secure_images = []

        for pk, b_image in db_records:
            ext = guess_image_ext(b_image)
            if ext is not None:
                plain_images.append((pk, b_image, ext))
                continue

            try:
                nonce, tag, ciphertext = b_image[:16], b_image[16:32], b_image[32:]
                cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
                plain = cipher.decrypt_and_verify(ciphertext, tag)
                ext = guess_image_ext(plain) or "png"
                secure_images.append((pk, plain, ext))
            except Exception as e:
                logger.warning(f"Failed to decrypt DB image {pk}: {e}")
                plain_images.append((pk, b_image, "raw"))

        return plain_images, secure_images

    def delete_db_image(self, pk: int) -> None:
        """Delete image entry from database by primary key ID."""
        self.db.delete_image(pk)

    def transfer_images_to_workspace(self, image_paths: list[str], target_folder: str) -> int:
        """Copy list of image file paths into target workspace folder."""
        dest = Path(target_folder)
        dest.mkdir(parents=True, exist_ok=True)
        copied = 0
        for src in image_paths:
            p = Path(src)
            if p.exists():
                copy(p, dest)
                copied += 1
        return copied
