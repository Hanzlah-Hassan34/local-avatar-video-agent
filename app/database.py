import sqlite3
from pathlib import Path


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS avatars (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    voice_path TEXT,
                    consent_path TEXT NOT NULL,
                    consent_text TEXT NOT NULL,
                    driving_video_path TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS voice_profiles (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    sample_path TEXT NOT NULL,
                    consent_path TEXT NOT NULL,
                    consent_text TEXT NOT NULL,
                    profile_path TEXT,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    script TEXT NOT NULL,
                    language TEXT NOT NULL,
                    style TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    avatar_id TEXT REFERENCES avatars(id),
                    voice_id TEXT REFERENCES voice_profiles(id),
                    video_path TEXT,
                    error TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS segments (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    audio_path TEXT,
                    video_path TEXT,
                    approved INTEGER NOT NULL DEFAULT 0,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    UNIQUE(job_id, position)
                );

                CREATE TABLE IF NOT EXISTS job_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    event TEXT NOT NULL,
                    detail TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS job_events_job_id_id
                ON job_events(job_id, id);
                """
            )
            self._add_column(connection, "jobs", "avatar_id", "TEXT REFERENCES avatars(id)")
            self._add_column(connection, "jobs", "voice_id", "TEXT REFERENCES voice_profiles(id)")
            self._add_column(connection, "jobs", "video_path", "TEXT")
            self._add_column(connection, "jobs", "error", "TEXT")
            self._add_column(
                connection, "jobs", "cancel_requested", "INTEGER NOT NULL DEFAULT 0"
            )
            self._add_column(connection, "segments", "audio_path", "TEXT")
            self._add_column(connection, "segments", "video_path", "TEXT")
            self._add_column(
                connection, "segments", "approved", "INTEGER NOT NULL DEFAULT 0"
            )
            self._add_column(
                connection, "segments", "attempts", "INTEGER NOT NULL DEFAULT 0"
            )
            self._add_column(connection, "segments", "error", "TEXT")
            self._make_avatar_voice_optional(connection)

    @staticmethod
    def _add_column(
        connection: sqlite3.Connection, table: str, column: str, definition: str
    ) -> None:
        columns = {
            row["name"] for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    @staticmethod
    def _make_avatar_voice_optional(connection: sqlite3.Connection) -> None:
        columns = {
            row["name"]: row
            for row in connection.execute("PRAGMA table_info(avatars)")
        }
        if not columns["voice_path"]["notnull"]:
            return

        connection.execute("PRAGMA foreign_keys = OFF")
        connection.executescript(
            """
            CREATE TABLE avatars_new (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                image_path TEXT NOT NULL,
                voice_path TEXT,
                consent_path TEXT NOT NULL,
                consent_text TEXT NOT NULL,
                driving_video_path TEXT,
                created_at TEXT NOT NULL
            );

            INSERT INTO avatars_new
            (id, name, image_path, voice_path, consent_path, consent_text,
             driving_video_path, created_at)
            SELECT id, name, image_path, voice_path, consent_path, consent_text,
                   driving_video_path, created_at
            FROM avatars;

            DROP TABLE avatars;
            ALTER TABLE avatars_new RENAME TO avatars;
            """
        )
        connection.execute("PRAGMA foreign_keys = ON")
