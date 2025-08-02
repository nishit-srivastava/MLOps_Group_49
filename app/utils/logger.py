import sqlite3
import os
from datetime import datetime

DB_PATH = "logs/predictions.db"

def init_db():
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                input TEXT,
                output REAL
            )
        ''')
        conn.commit()
        conn.close()

def log_prediction(input_data, prediction):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO predictions (timestamp, input, output) VALUES (?, ?, ?)',
              (datetime.utcnow().isoformat(), str(input_data), prediction))
    conn.commit()
    conn.close()

init_db()
