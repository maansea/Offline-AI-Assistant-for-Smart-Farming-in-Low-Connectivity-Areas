import os
import sqlite3
import tempfile
from pathlib import Path

from flask import Flask, g, jsonify, render_template_string, request, current_app

from src.recommendation_logic import RecommendationEngine
from voice_utils import transcribe_audio_file

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = os.environ.get("DATABASE", str(BASE_DIR / "farm_assistant.db"))

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Smart Farming Assistant</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; background: #f7f9f4; color: #1f2d1f; }
    .card { background: white; padding: 1.2rem; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 1rem; }
    input, select, button { padding: 0.7rem; margin: 0.25rem 0; width: 100%; border-radius: 8px; border: 1px solid #cfd9c9; }
    button { background: #2e7d32; color: white; cursor: pointer; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; }
    .muted { color: #5c6b5c; font-size: 0.95rem; }
  </style>
</head>
<body>
  <h1>🌾 Smart Farming Assistant</h1>
  <p class="muted">Offline-friendly crop and fertilizer guidance for rural farmers.</p>

  <div class="card">
    <h2>Farmer Input</h2>
    <form method="post" action="/recommend">
      <div class="grid">
        <div><label>Soil Type</label><select name="soil_type"><option>Loamy</option><option>Sandy</option><option>Clayey</option><option>Black</option><option>Red</option><option>Alluvial</option></select></div>
        <div><label>Season</label><select name="season"><option>Kharif</option><option>Rabi</option><option>Zaid</option><option>Summer</option><option>Winter</option><option>Monsoon</option></select></div>
        <div><label>N</label><input name="crop_n" value="90" /></div>
        <div><label>P</label><input name="crop_p" value="42" /></div>
        <div><label>K</label><input name="crop_k" value="43" /></div>
        <div><label>Temperature</label><input name="temperature" value="20.87" /></div>
        <div><label>Humidity</label><input name="humidity" value="82" /></div>
        <div><label>pH</label><input name="ph" value="6.5" /></div>
        <div><label>Rainfall</label><input name="rainfall" value="202.93" /></div>
        <div><label>Fertilizer Soil Type</label><select name="fertilizer_soil_type"><option>Sandy</option><option>Loamy</option><option>Clayey</option><option>Black</option><option>Red</option><option>Alluvial</option></select></div>
        <div><label>Crop Type</label><input name="fertilizer_crop_type" value="Maize" /></div>
        <div><label>Fertilizer Temperature</label><input name="fertilizer_temperature" value="26" /></div>
        <div><label>Humidity</label><input name="fertilizer_humidity" value="52" /></div>
        <div><label>Moisture</label><input name="fertilizer_moisture" value="38" /></div>
        <div><label>Nitrogen</label><input name="nitrogen" value="37" /></div>
        <div><label>Phosphorous</label><input name="phosphorous" value="0" /></div>
        <div><label>Potassium</label><input name="potassium" value="0" /></div>
      </div>
      <button type="submit">Get Recommendations</button>
    </form>
  </div>

  <div class="card">
    <h2>Voice Assistant</h2>
    <p class="muted">Use a microphone or upload a short audio clip for simple offline speech-to-text.</p>
    <p><a href="/voice">Open microphone page</a></p>
    <p><a href="/history">View Recommendation History</a></p>
  </div>
</body>
</html>
"""

VOICE_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Voice Input</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; background: #f7f9f4; color: #1f2d1f; }
    .card { background: white; padding: 1.2rem; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 1rem; }
    button, input { padding: 0.7rem; margin-top: 0.5rem; width: 100%; border-radius: 8px; border: 1px solid #cfd9c9; }
    button { background: #2e7d32; color: white; cursor: pointer; }
  </style>
</head>
<body>
  <div class="card">
    <h2>🎤 Voice Input</h2>
    <p>Upload a short WAV audio file to convert it to text using Vosk.</p>
    <form action="/voice" method="post" enctype="multipart/form-data">
      <input type="file" name="audio" accept=".wav,audio/wav" />
      <button type="submit">Transcribe</button>
    </form>
    <p id="result">{{ result }}</p>
    <p><a href="/">Back to main page</a></p>
  </div>
</body>
</html>
"""


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_mapping(
        DATABASE=DATABASE_PATH,
        SECRET_KEY="dev-secret-key",
    )

    if test_config is not None:
        app.config.update(test_config)

    @app.before_request
    def before_request():
        g.db = get_db()

    @app.teardown_appcontext
    def close_connection(exception):
        db = g.pop("db", None)
        if db is not None:
            db.close()

    @app.route("/")
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.route("/recommend", methods=["GET", "POST"])
    def recommend():
        if request.method == "GET":
            return render_template_string(HTML_TEMPLATE)

        crop_input = {
            "N": float(request.form.get("crop_n", 0)),
            "P": float(request.form.get("crop_p", 0)),
            "K": float(request.form.get("crop_k", 0)),
            "temperature": float(request.form.get("temperature", 0)),
            "humidity": float(request.form.get("humidity", 0)),
            "ph": float(request.form.get("ph", 0)),
            "rainfall": float(request.form.get("rainfall", 0)),
            "soil_type": request.form.get("soil_type", "Loamy"),
            "season": request.form.get("season", "Kharif"),
        }

        fertilizer_input = {
            "Temparature": float(request.form.get("fertilizer_temperature", 0)),
            "Humidity": float(request.form.get("fertilizer_humidity", 0)),
            "Moisture": float(request.form.get("fertilizer_moisture", 0)),
            "Soil Type": request.form.get("fertilizer_soil_type", "Sandy"),
            "Crop Type": request.form.get("fertilizer_crop_type", "Maize"),
            "Nitrogen": float(request.form.get("nitrogen", 0)),
            "Potassium": float(request.form.get("potassium", 0)),
            "Phosphorous": float(request.form.get("phosphorous", 0)),
        }

        engine = RecommendationEngine()
        result = engine.get_full_recommendation(crop_input, fertilizer_input)

        crop_name = result["crop_recommendation"].get("recommended_crop", "N/A")
        fertilizer_name = result["fertilizer_recommendation"].get("recommended_fertilizer", "N/A")

        g.db.execute(
            "INSERT INTO recommendations (crop_name, fertilizer_name, created_at) VALUES (?, ?, datetime('now'))",
            (crop_name, fertilizer_name),
        )
        g.db.commit()

        return jsonify(result)

    @app.route("/voice", methods=["GET", "POST"])
    def voice():
        result = ""
        if request.method == "POST":
            audio_file = request.files.get("audio")
            if audio_file and audio_file.filename:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    audio_file.save(temp_file.name)
                    result = transcribe_audio_file(temp_file.name)
                    os.unlink(temp_file.name)
            else:
                result = "No audio file was provided."
        return render_template_string(VOICE_PAGE, result=result)

    @app.route("/history")
    def history():
        rows = g.db.execute(
            "SELECT crop_name, fertilizer_name, created_at FROM recommendations ORDER BY id DESC LIMIT 10"
        ).fetchall()
        history_rows = [
            {"crop_name": row[0], "fertilizer_name": row[1], "created_at": row[2]}
            for row in rows
        ]
        return jsonify(history_rows)

    init_db(app)
    return app


def get_db():
    if "db" not in g:
        conn = sqlite3.connect(current_app.config["DATABASE"])
        conn.row_factory = sqlite3.Row
        g.db = conn
    return g.db


def init_db(app=None):
    if app is not None:
        db_path = app.config["DATABASE"]
    else:
        db_path = DATABASE_PATH

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crop_name TEXT NOT NULL,
            fertilizer_name TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, debug=True)
