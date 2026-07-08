import os
import sqlite3
import subprocess
import uuid
from pathlib import Path

from flask import Flask, g, jsonify, render_template_string, request, current_app

from src.recommendation_logic import RecommendationEngine
from voice_utils import MODEL_PATH, build_transcription_error_message, transcribe_audio_file, transcribe_microphone_stream

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
button:disabled { opacity: 0.6; cursor: not-allowed; }
.status { margin-top: 0.75rem; color: #4b6b3a; font-weight: 600; }
</style>
</head>
<body>
<div class="card">
<h2>🎤 Voice Input</h2>
<p>Record your voice and upload the audio file to the server for local transcription.</p>
<button id="startButton" type="button">Start Recording</button>
<button id="stopButton" type="button" disabled>Stop Recording</button>
<div id="status" class="status">Ready to record.</div>
<form action="/voice" method="post" enctype="multipart/form-data" style="margin-top: 1rem;">
<input type="file" name="audio" accept=".wav,audio/wav" />
<button type="submit">Transcribe Uploaded Audio</button>
</form>
<p id="result">{{ result }}</p>
<p><a href="/">Back to main page</a></p>
</div>
<script>
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const status = document.getElementById('status');
const result = document.getElementById('result');
let stream = null;
let mediaRecorder = null;

startButton.addEventListener('click', async () => {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    result.textContent = 'Microphone access is not supported in this browser.';
    status.textContent = 'Browser microphone support missing.';
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    const sampleRate = audioContext.sampleRate;

    function floatTo16BitPCM(output, input) {
      for (let i = 0; i < input.length; i++) {
        const s = Math.max(-1, Math.min(1, input[i]));
        output.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      }
    }
    function writeString(view, offset, string) {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    }
    function encodeWAV(samples, sampleRate) {
      const buffer = new ArrayBuffer(44 + samples.length * 2);
      const view = new DataView(buffer);
      writeString(view, 0, 'RIFF');
      view.setUint32(4, 36 + samples.length * 2, true);
      writeString(view, 8, 'WAVE');
      writeString(view, 12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      writeString(view, 36, 'data');
      view.setUint32(40, samples.length * 2, true);
      floatTo16BitPCM(new DataView(buffer, 44), samples);
      return new Blob([view], { type: 'audio/wav' });
    }

    let recordingBuffer = [];
    processor.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      recordingBuffer.push(new Float32Array(input));
    };
    source.connect(processor);
    processor.connect(audioContext.destination);

    mediaRecorder = {
      state: 'recording',
      stop: () => {
        source.disconnect();
        processor.disconnect();
        audioContext.close();

        const length = recordingBuffer.reduce((sum, chunk) => sum + chunk.length, 0);
        const merged = new Float32Array(length);
        let offset = 0;
        for (const chunk of recordingBuffer) {
          merged.set(chunk, offset);
          offset += chunk.length;
        }
        const wavBlob = encodeWAV(merged, sampleRate);
        const fileName = 'recording.wav';
        const file = new File([wavBlob], fileName, { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('audio', file, fileName);

        fetch('/voice', {
          method: 'POST',
          body: formData,
          headers: { 'X-Requested-With': 'XMLHttpRequest' },
        })
          .then(async (response) => {
            const payload = await response.json();
            result.textContent = payload.result || 'No transcription returned.';
            status.textContent = 'Recording uploaded and processed.';
          })
          .catch(() => {
            result.textContent = 'Recording upload failed.';
            status.textContent = 'Recording upload failed.';
          })
          .finally(() => {
            stream.getTracks().forEach((track) => track.stop());
            stream = null;
            startButton.disabled = false;
            stopButton.disabled = true;
          });
      },
    };

    status.textContent = 'Recording... speak now.';
    startButton.disabled = true;
    stopButton.disabled = false;
  } catch (error) {
    result.textContent = 'Unable to access the microphone.';
    status.textContent = 'Microphone access failed.';
  }
});

stopButton.addEventListener('click', () => {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
});
</script>
</body>
</html>
"""


def create_app(test_config=None):
    app = Flask(__name__)

    upload_folder = BASE_DIR / "uploads"
    upload_folder.mkdir(exist_ok=True)

    app.config.from_mapping(
        DATABASE=DATABASE_PATH,
        SECRET_KEY="dev-secret-key",
        UPLOAD_FOLDER=str(upload_folder),
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
        saved_path = ""

        if request.method == "POST":
            audio_file = request.files.get("audio")
            if audio_file and audio_file.filename:
                try:
                    upload_folder = Path(current_app.config["UPLOAD_FOLDER"])
                    upload_folder.mkdir(exist_ok=True)

                    suffix = Path(audio_file.filename).suffix or ".webm"
                    saved_path = upload_folder / f"{uuid.uuid4().hex}{suffix}"
                    audio_file.save(saved_path)

                    converted_path = None
                    try:
                        converted_path = str(saved_path) + ".wav"

                        # FIX: force mono (-ac 1), 16000 Hz (-ar 16000), 16-bit PCM (pcm_s16le)
                        # so that Vosk can always read the file regardless of the recording device.
                        subprocess.run(
                            [
                                "ffmpeg",
                                "-y",
                                "-i", str(saved_path),
                                "-ar", "16000",   # resample to 16 kHz
                                "-ac", "1",        # downmix to mono
                                "-sample_fmt", "s16",  # 16-bit PCM
                                converted_path,
                            ],
                            check=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                        )

                        result = transcribe_audio_file(converted_path)
                        if not result:
                            result = build_transcription_error_message(converted_path, MODEL_PATH)

                    except Exception:
                        result = build_transcription_error_message(str(saved_path), MODEL_PATH)
                    finally:
                        if converted_path and os.path.exists(converted_path):
                            os.unlink(converted_path)

                except Exception:
                    result = "Audio upload failed."
            else:
                result = "No audio file was provided."

        if request.method == "POST" and request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"result": result, "saved_path": str(saved_path)})

        return render_template_string(VOICE_PAGE, result=result)

    @app.route("/voice/offline", methods=["POST"])
    def voice_offline():
        duration = int(request.form.get("duration", 5))
        result = transcribe_microphone_stream(duration=duration)
        return jsonify({"result": result})

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
