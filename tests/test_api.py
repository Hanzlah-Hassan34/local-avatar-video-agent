import sqlite3

from fastapi.testclient import TestClient

from app.database import Database
from app.main import create_app


def test_health_and_studio_are_available(tmp_path):
    with TestClient(create_app(tmp_path / "jobs.sqlite3")) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        assert "Local Avatar Studio" in client.get("/").text


def test_create_job_persists_script_and_segments(tmp_path):
    with TestClient(create_app(tmp_path / "jobs.sqlite3")) as client:
        response = client.post(
            "/api/jobs",
            json={
                "script": "First sentence. Second sentence.",
                "language": "en",
                "style": "natural",
            },
        )

        assert response.status_code == 201
        job = response.json()
        assert job["status"] == "draft"
        assert [segment["text"] for segment in job["segments"]] == [
            "First sentence. Second sentence."
        ]
        assert client.get("/api/jobs").json()[0]["id"] == job["id"]


def test_create_job_rejects_blank_script(tmp_path):
    with TestClient(create_app(tmp_path / "jobs.sqlite3")) as client:
        response = client.post("/api/jobs", json={"script": "   "})

    assert response.status_code == 422


def test_unknown_job_returns_not_found(tmp_path):
    with TestClient(create_app(tmp_path / "jobs.sqlite3")) as client:
        response = client.get("/api/jobs/unknown")

    assert response.status_code == 404


def test_system_status_reports_runtime_capability(tmp_path):
    with TestClient(create_app(tmp_path / "jobs.sqlite3")) as client:
        response = client.get("/api/system")

    assert response.status_code == 200
    assert isinstance(response.json()["cuda_available"], bool)


def test_system_status_without_torch_reports_cuda_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.importlib.util.find_spec", lambda name: None)
    with TestClient(create_app(tmp_path / "jobs.sqlite3")) as client:
        response = client.get("/api/system")

    assert response.status_code == 200
    assert response.json()["torch_installed"] is False
    assert response.json()["cuda_available"] is False
    assert response.json()["voice_pipeline_configured"] is False
    assert response.json()["video_pipeline_configured"] is False
    assert response.json()["full_pipeline_configured"] is False
    assert response.json()["gpu_worker_stages_configured"] is False


def test_system_status_distinguishes_configured_pipeline_stages(tmp_path):
    config_path = tmp_path / "workers.json"
    config_path.write_text(
        '{"voice_clone": {"command": ["worker"]}, '
        '"tts": {"command": ["worker"]}, '
        '"motion": {"command": ["worker"]}}',
        encoding="utf-8",
    )
    with TestClient(
        create_app(
            tmp_path / "jobs.sqlite3",
            data_dir=tmp_path,
            worker_config_path=config_path,
        )
    ) as client:
        status = client.get("/api/system").json()

    assert status["voice_pipeline_configured"] is True
    assert status["video_pipeline_configured"] is False
    assert status["full_pipeline_configured"] is False
    assert status["missing_workers"] == ["lip_sync", "post", "qc"]


def test_cancel_draft_finishes_immediately(tmp_path):
    with TestClient(create_app(tmp_path / "jobs.sqlite3")) as client:
        job = client.post("/api/jobs", json={"script": "Draft only."}).json()
        response = client.post(f"/api/jobs/{job['id']}/cancel")
        cancelled = client.get(f"/api/jobs/{job['id']}").json()

    assert response.status_code == 202
    assert response.json()["status"] == "cancelled"
    assert cancelled["status"] == "cancelled"
    assert cancelled["can_cancel"] is False


def test_cancel_video_quality_review_finishes_immediately(tmp_path):
    with TestClient(create_app(tmp_path / "jobs.sqlite3")) as client:
        job = client.post("/api/jobs", json={"script": "Review this avatar."}).json()
        with client.app.state.database.connect() as connection:
            connection.execute(
                "UPDATE jobs SET status = 'video_review' WHERE id = ?", (job["id"],)
            )
        response = client.post(f"/api/jobs/{job['id']}/cancel")
        cancelled = client.get(f"/api/jobs/{job['id']}").json()

    assert response.status_code == 202
    assert response.json()["status"] == "cancelled"
    assert cancelled["status"] == "cancelled"
    assert cancelled["can_cancel"] is False


def test_startup_marks_interrupted_jobs_retryable(tmp_path):
    database_path = tmp_path / "jobs.sqlite3"
    with TestClient(create_app(database_path, data_dir=tmp_path)) as client:
        job = client.post("/api/jobs", json={"script": "Resume after restart."}).json()
        with client.app.state.database.connect() as connection:
            connection.execute(
                "UPDATE jobs SET status = 'motion' WHERE id = ?", (job["id"],)
            )

    with TestClient(create_app(database_path, data_dir=tmp_path)) as client:
        recovered = client.get(f"/api/jobs/{job['id']}").json()

    assert recovered["status"] == "failed"
    assert "App restarted" in recovered["error"]


def test_database_migration_makes_avatar_voice_optional(tmp_path):
    database_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE avatars (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, image_path TEXT NOT NULL,
                voice_path TEXT NOT NULL, consent_path TEXT NOT NULL,
                consent_text TEXT NOT NULL, driving_video_path TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY, script TEXT NOT NULL, language TEXT NOT NULL,
                style TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, avatar_id TEXT REFERENCES avatars(id)
            );
            CREATE TABLE segments (
                id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                position INTEGER NOT NULL, text TEXT NOT NULL, status TEXT NOT NULL,
                UNIQUE(job_id, position)
            );
            INSERT INTO avatars VALUES
                ('legacy-avatar', 'Legacy', 'image.png', 'voice.wav', 'consent.wav',
                 'consent statement with enough characters', NULL, '2026-01-01');
            INSERT INTO jobs VALUES
                ('legacy-job', 'Hello.', 'en', 'natural', 'draft', '2026-01-01',
                 '2026-01-01', 'legacy-avatar');
            """
        )

    database = Database(database_path)
    database.initialize()

    with database.connect() as connection:
        avatar_columns = {
            row["name"]: row
            for row in connection.execute("PRAGMA table_info(avatars)")
        }
        foreign_key_violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        connection.execute(
            """
            INSERT INTO avatars
            (id, name, image_path, consent_path, consent_text, created_at)
            VALUES ('new-avatar', 'New', 'new.png', 'consent.wav',
                    'face consent statement', '2026-01-01')
            """
        )

    assert avatar_columns["voice_path"]["notnull"] == 0
    assert foreign_key_violations == []


def test_startup_marks_interrupted_voice_clones_retryable(tmp_path):
    database_path = tmp_path / "jobs.sqlite3"
    database = Database(database_path)
    database.initialize()
    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO voice_profiles
            (id, name, sample_path, consent_path, consent_text, status, created_at, updated_at)
            VALUES ('voice-1', 'Voice', 'sample.wav', 'consent.wav', 'consent text',
                    'cloning', '2026-01-01', '2026-01-01')
            """
        )

    with TestClient(create_app(database_path, data_dir=tmp_path)) as client:
        profile = client.get("/api/voices").json()[0]

    assert profile["status"] == "failed"
    assert "App restarted during voice cloning" in profile["error"]
