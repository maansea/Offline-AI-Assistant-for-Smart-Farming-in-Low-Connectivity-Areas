import io
import os
import tempfile

import pytest

from app import create_app, get_db


@pytest.fixture()
def client():
    db_fd, db_path = tempfile.mkstemp()
    app = create_app({
        "TESTING": True,
        "DATABASE": db_path,
    })
    with app.test_client() as client:
        with app.app_context():
            from app import init_db
            init_db()
        yield client

    os.close(db_fd)
    os.unlink(db_path)


def test_home_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Smart Farming Assistant" in response.data


def test_recommendation_endpoint_returns_results(client):
    response = client.post(
        "/recommend",
        data={
            "crop_n": "90",
            "crop_p": "42",
            "crop_k": "43",
            "temperature": "20.87",
            "humidity": "82",
            "ph": "6.5",
            "rainfall": "202.93",
            "soil_type": "Loamy",
            "season": "Kharif",
            "fertilizer_temperature": "26",
            "fertilizer_humidity": "52",
            "fertilizer_moisture": "38",
            "fertilizer_soil_type": "Sandy",
            "fertilizer_crop_type": "Maize",
            "nitrogen": "37",
            "potassium": "0",
            "phosphorous": "0",
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert "crop_recommendation" in payload
    assert "fertilizer_recommendation" in payload


def test_voice_page_includes_microphone_controls(client):
    response = client.get("/voice")
    assert response.status_code == 200
    assert b"Start Recording" in response.data
    assert b"Stop Recording" in response.data


def test_voice_endpoint_handles_invalid_audio(client):
    response = client.post(
        "/voice",
        data={"audio": (io.BytesIO(b"not-a-valid-wav"), "recording.wav")},
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert "result" in payload
    assert payload["result"]


def test_voice_endpoint_returns_helpful_message_when_model_missing(client):
    response = client.post(
        "/voice",
        data={"audio": (io.BytesIO(b"not-a-valid-wav"), "recording.wav")},
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert "Offline speech model" in payload["result"]


def test_voice_offline_endpoint_returns_json(client):
    response = client.post(
        "/voice/offline",
        data={"duration": "1"},
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert "result" in payload


def test_history_page_lists_recent_entries(client):
    with client.application.app_context():
        conn = get_db()
        conn.execute(
            "INSERT INTO recommendations (crop_name, fertilizer_name, created_at) VALUES (?, ?, datetime('now'))",
            ("Rice", "Urea"),
        )
        conn.commit()

    response = client.get("/history")
    assert response.status_code == 200
    assert b"Rice" in response.data
