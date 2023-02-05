import sys
from hashlib import sha256
from os.path import join
from sqlite3 import connect


def call_db(call, data=None):
    # Create db
    if hasattr(sys, "_MEIPASS"):
        path = join(sys._MEIPASS + "app.db")
        conn = connect(path)
    else:
        conn = connect("app.db")

    # Create cursor
    c = conn.cursor()

    # Execute SQL command
    if data is None:
        c.execute(call)
    else:
        c.execute(call, data)
    records = c.fetchall()

    # Commit changes
    conn.commit()

    # Close connection
    conn.close()

    return records


def get_sha(text):
    enc = sha256()
    enc.update(text.encode("utf-8"))
    return enc.hexdigest()


def extend_key(text):
    if len(text) < 16:
        while len(text) < 16:
            text += text
    return text[:16].encode("utf-8")
