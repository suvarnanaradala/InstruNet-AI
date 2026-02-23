from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, session
)
from flask import send_file
from flask import redirect, url_for
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import librosa.display
import numpy as np
import os
import uuid
from werkzeug.utils import secure_filename
from torchvision import models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import datetime

# ================= APP =================
app = Flask(__name__)
# @app.before_request
# def clear_session_once():
#     if not hasattr(app, "_session_cleared"):
#         session.clear()
#         app._session_cleared = True

app.secret_key = "instrunet-temp-history"

UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {"wav", "mp3", "flac"}
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LOAD MODEL =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_resnet18_irmas.pth")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 11)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ================= LABELS =================
INSTRUMENTS = [
    "Cello","Clarinet","Flute","Acoustic Guitar","Electric Guitar",
    "Organ","Piano","Saxophone","Trumpet","Violin","Voice"
]

# ================= UTILS =================
def allowed_file(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# ================= PAGES =================
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/analysis")
def analysis_page():
    return render_template("analysis.html")

@app.route("/results")
def results_page():
    history = session.get("history", [])

    if history:
        latest = history[0]
        return render_template(
            "results.html",
            detected=latest.get("detected", []),
            audio_path=latest.get("file_path"),
            mel_filename=latest.get("filename")   # ⭐ ADD THIS
        )

    return render_template(
        "results.html",
        detected=[],
        audio_path=None,
        mel_filename=None
    )



@app.route("/history")
def history_page():
    history = session.get("history", [])
    return render_template("history.html", history=history[:5])


@app.route("/about")
def about_page():
    return render_template("about.html")


# ================= CLEAR TEMP HISTORY =================
@app.route("/clear_history")
def clear_history():
    session.pop("history", None)
    session.pop("last_original_name", None)
    return redirect(url_for("history_page"))


# ================= UPLOAD =================
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("audio")
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    original_name = secure_filename(file.filename)
    filename = f"{uuid.uuid4()}_{original_name}"

    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    # store original name temporarily
    session["last_original_name"] = original_name

    return jsonify({"filename": filename})

@app.route("/uploads/<filename>")
def serve_audio(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ================= ANALYZE =================
@app.route("/analyze", methods=["POST"])
def analyze():
    filename = request.json.get("filename")
    if not filename:
        return jsonify({"error": "Missing filename"}), 400

    path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        y, sr = librosa.load(path, sr=22050, mono=True)
    except Exception:
        return jsonify({"error": "Failed to decode audio"}), 400

    if len(y) < sr:
        return jsonify({"error": "Audio too short (min 1s)"}), 400

    window_sec, hop_sec = 3.0, 1.0
    seg_len, hop_len = int(window_sec * sr), int(hop_sec * sr)

    if len(y) < seg_len:
        y = np.pad(y, (0, seg_len - len(y)))

    segments = []

    for start in range(0, len(y) - seg_len + 1, hop_len):
        seg = y[start:start + seg_len]

        rms = np.mean(librosa.feature.rms(y=seg))
        if rms < 1e-4:
            continue

        mel = librosa.feature.melspectrogram(
            y=seg, sr=sr, n_mels=128,
            n_fft=2048, hop_length=512, fmax=8000
        )
        mel = librosa.power_to_db(mel, ref=np.max)

        mel = mel[:, :128] if mel.shape[1] >= 128 else \
              np.pad(mel, ((0,0),(0,128-mel.shape[1])))

        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        segments.append(mel)

        if len(segments) >= 120:
            break

    if not segments:
        return jsonify({"error": "No meaningful audio detected"}), 400

    x = torch.from_numpy(np.stack(segments)).unsqueeze(1).float()
    x = F.interpolate(x, size=(224,224), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()

    timeline = [
        {"time": round(i * hop_sec, 2), "probs": p.tolist()}
        for i, p in enumerate(probs)
    ]

    if len(timeline) > 60:
        timeline = timeline[::2]

    avg = np.mean([t["probs"] for t in timeline], axis=0)
    top_idx = int(np.argmax(avg))

# Build sorted probabilities (ONLY ONCE)
    probs_dict = {
    INSTRUMENTS[i]: float(avg[i])
    for i in range(len(INSTRUMENTS))
}

    sorted_probs = sorted(
    probs_dict.items(),
    key=lambda x: x[1],
    reverse=True
)

# ================= HISTORY =================
    if "history" not in session:
        session["history"] = []

    original_name = session.get("last_original_name", filename)

# remove old entry of same song
    session["history"] = [
    h for h in session["history"]
    if h["original_name"] != original_name
]

    session["history"].insert(0, {
    "id": str(uuid.uuid4()),
    "filename": filename,
    "original_name": original_name,
    "prediction": INSTRUMENTS[top_idx],
    "confidence": float(avg[top_idx]),
    "file_path": f"uploads/{filename}",
    "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "detected": sorted_probs[:5]
})

# limit history
    session["history"] = session["history"][:10]
    session.modified = True

    return jsonify({
    "filename": filename,
    "average": probs_dict,
    "timeline": timeline
})



# ================= VIEW OLD RESULT =================
@app.route("/results/<rid>")
def view_old_result(rid):
    for h in session.get("history", []):
        if h["id"] == rid:
            return render_template(
                "results.html",
                filename=h["original_name"],
                prediction=h["prediction"],
                confidence=h["confidence"],
                audio_path=h["file_path"],
                mel_filename=h["filename"]
            )
    return "Result not found", 404

# ================= MEL SPECTROGRAM =================
@app.route("/mel_spectrogram", methods=["POST"])
def mel_spectrogram():
    filename = request.json.get("filename")
    path = os.path.join(UPLOAD_FOLDER, filename)

    y, sr = librosa.load(path, sr=22050, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128,
        n_fft=2048, hop_length=512, fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(9, 4))
    librosa.display.specshow(mel_db, sr=sr, hop_length=512, cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    plt.close()
    buf.seek(0)

    return jsonify({"image": base64.b64encode(buf.read()).decode()})

# ================= DOWNLOAD MEL SPECTROGRAM =================
@app.route("/download_mel/<filename>")
def download_mel(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)

    y, sr = librosa.load(path, sr=22050, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128,
        n_fft=2048, hop_length=512, fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(9, 4))
    librosa.display.specshow(mel_db, sr=sr, hop_length=512, cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)

    return send_file(
        buf,
        mimetype="image/png",
        as_attachment=True,
        download_name=f"{filename}_mel.png"
    )


# ================= EXPORT PDF =================
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import KeepTogether
from reportlab.platypus import PageBreak
from reportlab.platypus import Preformatted
from reportlab.platypus import Table
from reportlab.platypus import TableStyle
from reportlab.platypus import HRFlowable
from reportlab.platypus import FrameBreak
from reportlab.platypus import NextPageTemplate
from reportlab.platypus import Frame
from reportlab.platypus import PageTemplate
from reportlab.platypus import BaseDocTemplate
from reportlab.platypus import Flowable
from reportlab.platypus import ListItem
from reportlab.platypus import ListFlowable
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Spacer
from reportlab.platypus import Paragraph
from reportlab.platypus import Image
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.pagesizes import A4

@app.route("/export_pdf")
def export_pdf():
    history = session.get("history", [])
    if not history:
        return "No analysis to export", 400

    latest = history[0]

    file_path = os.path.join(UPLOAD_FOLDER, latest["filename"])

    pdf_path = os.path.join("exports", f"{latest['filename']}.pdf")
    os.makedirs("exports", exist_ok=True)

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>InstruNet AI - Analysis Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"<b>File:</b> {latest['original_name']}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Prediction:</b> {latest['prediction']}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Confidence:</b> {latest['confidence']*100:.1f}%", styles["Normal"]))

# ✅ ADD THIS LINE HERE
    elements.append(Paragraph(
    f"<b>Generated On:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    styles["Normal"]
))

    elements.append(Spacer(1, 0.3 * inch))


    elements.append(Spacer(1, 0.4 * inch))
    elements.append(Paragraph("<b>Top 5 Instruments:</b>", styles["Heading3"]))
    elements.append(Spacer(1, 0.2 * inch))

# Create table data
    data = [["Instrument", "Probability (%)"]]

    for inst, prob in latest["detected"]:
        data.append([inst, f"{prob*100:.1f}%"])

    table = Table(data, colWidths=[3*inch, 2*inch])
    table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("ALIGN", (1,1), (-1,-1), "RIGHT"),
]))

    elements.append(table)

    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph("<b>Model Information:</b>", styles["Heading3"]))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("Architecture: ResNet18", styles["Normal"]))
    elements.append(Paragraph("Input: Mel Spectrogram (128x128)", styles["Normal"]))
    elements.append(Paragraph("Classes: 11 Instruments", styles["Normal"]))


    # ================= MEL SPECTROGRAM IMAGE =================
    y, sr = librosa.load(file_path, sr=22050, mono=True)

    mel = librosa.feature.melspectrogram(
    y=y, sr=sr,
    n_mels=128,
    n_fft=2048,
    hop_length=512,
    fmax=8000
)

    mel_db = librosa.power_to_db(mel, ref=np.max)

    img_path = os.path.join("exports", "temp_mel.png")

    plt.figure(figsize=(6, 3))
    librosa.display.specshow(mel_db, sr=sr, hop_length=512, cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()

    elements.append(Spacer(1, 0.4 * inch))
    elements.append(Paragraph("<b>Mel Spectrogram:</b>", styles["Heading3"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Image(img_path, width=5*inch, height=2.5*inch))

    doc.build(elements)

    return send_from_directory("exports", f"{latest['filename']}.pdf", as_attachment=True)


# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
