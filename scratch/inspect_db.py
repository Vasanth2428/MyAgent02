import sqlite3
import os

db_path = "checkpoints/checkpoints.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print("Tables:", tables)
    for table in tables:
        tname = table[0]
        cursor.execute(f"PRAGMA table_info({tname})")
        cols = cursor.fetchall()
        print(f"Table {tname}:", [c[1] for c in cols])
else:
    print("Database file does not exist at:", db_path)
