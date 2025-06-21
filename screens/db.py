from utils import call_db


class DB:
    def __init__(self):
        pass

    @staticmethod
    def create_customers_table():
        call_db(
            """
        CREATE TABLE IF NOT EXISTS customers (
            name text
        ) """
        )

    @staticmethod
    def insert_customer(b_encoded_text):
        call_db(f"INSERT INTO customers VALUES ('{b_encoded_text}')")

    @staticmethod
    def get_customers():
        return call_db("SELECT * FROM customers")

    @staticmethod
    def delete_customer(b_encoded_text):
        call_db(f"DELETE FROM customers WHERE name='{b_encoded_text}'")

    @staticmethod
    def update_customer(new_encrypted, old_encrypted):
        call_db(
            f"UPDATE customers SET name='{new_encrypted}' WHERE name='{old_encrypted}'"
        )

    @staticmethod
    def create_images_table():
        call_db(
            """
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image blob
        ) """
        )

    @staticmethod
    def insert_image(blob_data):
        call_db("INSERT INTO images (image) VALUES (?)", [blob_data])

    @staticmethod
    def get_images():
        return call_db("SELECT * FROM images")

    @staticmethod
    def delete_image(key):
        call_db(f"DELETE FROM images WHERE id={key}")

    @staticmethod
    def create_configs_table():
        call_db(
            """
        CREATE TABLE IF NOT EXISTS configs (
            name text unique,
            value text
        ) """
        )

    @staticmethod
    def get_latest_detection_project():
        return call_db("SELECT value FROM configs WHERE name='latest_detection_project'")

    @staticmethod
    def set_latest_detection_project(active_project):
        call_db(
            f"INSERT OR REPLACE INTO configs VALUES "
            f"('latest_detection_project', '{active_project}')"
        )

    @staticmethod
    def create_passwords_table():
        call_db(
            """
        CREATE TABLE IF NOT EXISTS passwords (
            destination text,
            password text
        ) """
        )

    @staticmethod
    def get_login_password():
        return call_db("SELECT * FROM passwords WHERE destination='login'")

    @staticmethod
    def set_login_password(enc_pass):
        call_db(f"INSERT INTO passwords VALUES ('login', '{enc_pass}')")
