from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


def test_questions():
    r = client.get("/questions")
    assert r.status_code == 200
    data = r.json()
    assert "questions" in data and len(data["questions"]) == 6


def test_calculate_aura_requires_api_key(monkeypatch):
    # Ensure no keys configured (deny non-local in app code), testclient uses 127.0.0.1 so endpoint should still enforce header
    from app import config as config_mod
    def fake_settings():
        return config_mod.Settings(api_keys_csv=None)
    monkeypatch.setattr(config_mod, "get_settings", fake_settings)
    payload = {
        "q1": "Calm",
        "q2": "Lead",
        "q3": "Explain clearly",
        "q4": "Energized",
        "q5": "Often",
        "q6": "Always",
    }
    r = client.post("/calculate-aura", json=payload)
    assert r.status_code == 401


def test_calculate_aura_with_key(monkeypatch):
    # Allow all keys via monkeypatching settings getter
    from app import config as config_mod

    def fake_settings():
        return config_mod.Settings(openai_api_key="dummy", api_keys_csv="testkey", rate_limit_per_minute=600)

    monkeypatch.setattr(config_mod, "get_settings", fake_settings)

    payload = {
        "q1": "Calm",
        "q2": "Lead",
        "q3": "Explain clearly",
        "q4": "Energized",
        "q5": "Often",
        "q6": "Always",
    }
    r = client.post("/calculate-aura", json=payload, headers={"x-api-key": "testkey"})
    assert r.status_code == 200
    body = r.json()
    assert "aura_score" in body and "sub_scores" in body


def test_audio_endpoint_mocked_whisper(monkeypatch, tmp_path):
    # Monkeypatch settings and whisper transcription to avoid real API calls
    from app import config as config_mod
    from app import main as main_mod

    def fake_settings():
        return config_mod.Settings(openai_api_key="dummy", api_keys_csv="testkey", rate_limit_per_minute=600)

    def fake_transcribe(file):
        # Return deterministic transcript with some fillers, long words, punctuation
        return "Hello, um I am introducing myself. I enjoy collaborative problem-solving! Do you?"

    monkeypatch.setattr(config_mod, "get_settings", fake_settings)
    monkeypatch.setattr(main_mod, "_transcribe_with_whisper", fake_transcribe)

    # Create a fake MP3 file
    fake_audio = tmp_path / "intro.mp3"
    fake_audio.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x21fake")

    files = {"audio": ("intro.mp3", fake_audio.read_bytes(), "audio/mpeg")}
    data = {
        "q1": "Calm",
        "q2": "Lead",
        "q3": "Explain clearly",
        "q4": "Energized",
        "q5": "Often",
        "q6": "Always",
    }
    r = client.post("/calculate-aura-audio", files=files, data=data, headers={"x-api-key": "testkey"})
    assert r.status_code == 200
    body = r.json()
    assert "aura_score" in body and "audio_sub_scores" in body and "score_description" in body

