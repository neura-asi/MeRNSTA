# storage/db.py
import contextlib
import os
import sqlite3

import numpy as np
try:
    import psycopg
except ImportError:
    try:
        import psycopg2 as psycopg
    except ImportError:
        # Fallback for testing
        psycopg = None
import yaml

cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")
cfg = yaml.safe_load(open(cfg_path))
URI = cfg.get("storage_uri", "sqlite:///cortex.db")


class DBWrapper:
    def __init__(self):
        self.is_pg = URI.startswith("postgres")
        if self.is_pg and psycopg is not None:
            self.conn = psycopg.connect(URI)
            self.cur = self.conn.cursor()
            self.cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.cur.execute(
                "ALTER TABLE IF EXISTS mem ALTER COLUMN emb TYPE vector(384) USING emb::vector;"
            )
            self.cur.execute(
                """
                CREATE TABLE IF NOT EXISTS mem(
                  id   SERIAL PRIMARY KEY,
                  tok  TEXT,
                  emb  vector(384),
                  rank DOUBLE PRECISION DEFAULT 1,
                  ts   DOUBLE PRECISION
                )
            """
            )
            self.cur.execute(
                """
                CREATE INDEX IF NOT EXISTS mem_emb_hnsw
                ON mem USING hnsw (emb vector_l2_ops)
                WITH (m = 16, ef_construction = 64)
            """
            )
            self.conn.commit()
            self.placeholder = "%s"
            # Add code_evolution table for code evolution history
            self.cur.execute(
                """
                CREATE TABLE IF NOT EXISTS code_evolution(
                  id SERIAL PRIMARY KEY,
                  file_path TEXT,
                  goal TEXT,
                  patch TEXT,
                  applied BOOLEAN,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            self.conn.commit()
            self.cur.execute(
                """
                CREATE TABLE IF NOT EXISTS facts(
                  id   SERIAL PRIMARY KEY,
                  subject TEXT,
                  predicate TEXT,
                  object TEXT,
                  embedding TEXT,
                  media_type TEXT,
                  user_profile_id TEXT,
                  session_id TEXT,
                  subject_cluster_id TEXT,
                  confidence REAL,
                  contradiction_score REAL,
                  volatility_score REAL
                )
            """
            )
            # Add episodes table with session_id and user_profile_id
            self.cur.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes(
                  id SERIAL PRIMARY KEY,
                  start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  end_time TIMESTAMP,
                  subject_count INTEGER DEFAULT 0,
                  fact_count INTEGER DEFAULT 0,
                  summary TEXT,
                  session_id TEXT,
                  user_profile_id TEXT
                )
            """
            )
        else:
            path = URI.split(":///")[1]
            self.conn = sqlite3.connect(path, check_same_thread=False)
            self.cur = self.conn
            self.cur.execute(
                """
                CREATE TABLE IF NOT EXISTS mem(
                  id   INTEGER PRIMARY KEY,
                  tok  TEXT,
                  emb  BLOB,
                  rank REAL DEFAULT 1,
                  ts   REAL
                )
            """
            )
            # Add code_evolution table for code evolution history
            self.cur.execute(
                """
                CREATE TABLE IF NOT EXISTS code_evolution(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  file_path TEXT,
                  goal TEXT,
                  patch TEXT,
                  applied BOOLEAN,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            self.conn.commit()
            self.cur.execute(
                """
                CREATE TABLE IF NOT EXISTS facts(
                  id   INTEGER PRIMARY KEY,
                  subject TEXT,
                  predicate TEXT,
                  object TEXT,
                  embedding TEXT,
                  media_type TEXT,
                  user_profile_id TEXT,
                  session_id TEXT,
                  subject_cluster_id TEXT,
                  confidence REAL,
                  contradiction_score REAL,
                  volatility_score REAL
                )
            """
            )
            # Add episodes table with session_id and user_profile_id
            self.cur.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  start_time TEXT DEFAULT CURRENT_TIMESTAMP,
                  end_time TEXT,
                  subject_count INTEGER DEFAULT 0,
                  fact_count INTEGER DEFAULT 0,
                  summary TEXT,
                  session_id TEXT,
                  user_profile_id TEXT
                )
            """
            )
            self.conn.commit()
            self.placeholder = "?"

    # generic execute that swaps placeholders
    def execute(self, sql, params=None):
        if params is None:
            params = ()
        if self.is_pg:
            sql = sql.replace("?", self.placeholder)
        return self.cur.execute(sql, params)

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()


@contextlib.contextmanager
def db():
    wrapper = DBWrapper()
    try:
        yield wrapper
    finally:
        wrapper.commit()
        wrapper.close()


def initialize_db():
    import sqlite3
    import os
    from config.settings import DATABASE_CONFIG
    db_path = DATABASE_CONFIG.get('default_path', 'memory.db')
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY,
            subject TEXT,
            predicate TEXT,
            object TEXT,
            embedding TEXT,
            media_type TEXT,
            user_profile_id TEXT,
            session_id TEXT,
            subject_cluster_id TEXT,
            confidence REAL,
            contradiction_score REAL,
            volatility_score REAL
        )
    """)
    conn.commit()
    conn.close()
