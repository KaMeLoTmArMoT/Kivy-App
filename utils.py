import hashlib
import sqlite3
import sys
import os


def call_db(call, data=None):
    # Create db
    if hasattr(sys, '_MEIPASS'):
        path = os.path.join(sys._MEIPASS + 'app.db')
        conn = sqlite3.connect(path)
    else:
        conn = sqlite3.connect('app.db')

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
    enc = hashlib.sha256()
    enc.update(text.encode('utf-8'))
    return enc.hexdigest()


def extend_key(text):
    if len(text) < 16:
        while len(text) < 16:
            text += text
    return text[:16].encode('utf-8')