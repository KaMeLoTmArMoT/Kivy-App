import hashlib
import sqlite3


def call_db(call):
    # Create db
    conn = sqlite3.connect('app.db')

    # Create cursor
    c = conn.cursor()

    # Execute SQL command
    c.execute(call)
    records = c.fetchall()

    # Commit changes
    conn.commit()

    # Close connection
    conn.close()

    return records


def get_sha(text):
    enc = hashlib.sha256()
    enc.update(text.encode('utf-8'))
    return enc.hexdigest()


def extend_key(text):
    if len(text) < 16:
        while len(text) < 16:
            text += text
    return text[:16].encode('utf-8')