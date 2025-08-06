import logging
import sys
import json
import sqlite3
from datetime import datetime
import os

DB_PATH = "logs.db"

# Ensure DB exists and table is initialized
def init_db():
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                level TEXT,
                message TEXT
            )
        ''')
        conn.commit()
        conn.close()

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage()
        }
        return json.dumps(log_record)

class SQLiteHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        log_data = json.loads(log_entry)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO logs (timestamp, level, message) VALUES (?, ?, ?)",
            (log_data["timestamp"], log_data["level"], log_data["message"])
        )
        conn.commit()
        conn.close()

def get_logger(name="inference-logger"):
    init_db()

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = JsonFormatter()

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # SQLite handler
    sqlite_handler = SQLiteHandler()
    sqlite_handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(stream_handler)
        logger.addHandler(sqlite_handler)

    return logger
