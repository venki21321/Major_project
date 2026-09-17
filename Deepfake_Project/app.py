import os
from uuid import uuid4
from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.utils import secure_filename
from models.predict import predict_image
from models.news_predict import predict_news, predict_news_batch
from models.pdf_text import extract_pdf_text
from models.video_predict import predict_video

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Predict page
@app.route("/predict", methods=["POST"])
def predict():
    # Check if a file was uploaded
    if "file" not in request.files:
        return "No file selected."

    file = request.files["file"]

    # Check if filename is empty
    if file.filename == "":
        return "Please select a file."

    # Save uploaded file
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Predict and create a heatmap of the regions that influenced the decision.
    heatmap_filename = f"{uuid4().hex}.jpg"
    heatmap_path = os.path.join("static", "heatmaps", heatmap_filename)
    result = predict_image(filepath, heatmap_path)

    # Display result
    return render_template(
        "index.html",
        prediction=result,
        filename=file.filename,
        heatmap_filename=heatmap_filename
    )


@app.route("/api/predict-image", methods=["POST"])
def predict_image_api():
    """JSON image-prediction endpoint used by the Android application."""
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "Please select an image."}), 400

    file = request.files["file"]
    original_name = secure_filename(file.filename) or "upload.jpg"
    stored_name = f"{uuid4().hex}_{original_name}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
    file.save(filepath)

    heatmap_filename = f"{uuid4().hex}.jpg"
    heatmap_path = os.path.join("static", "heatmaps", heatmap_filename)
    try:
        result = predict_image(filepath, heatmap_path)
    except Exception as error:
        return jsonify({"error": str(error)}), 500

    return jsonify({
        "prediction": result,
        "filename": original_name,
        "heatmap_url": url_for(
            "static", filename=f"heatmaps/{heatmap_filename}", _external=True),
    })


@app.route("/predict-video", methods=["POST"])
def predict_video_route():
    if "video" not in request.files or request.files["video"].filename == "":
        return render_template("index.html", video_prediction="Please select a video.")

    file = request.files["video"]
    original_name = secure_filename(file.filename) or "upload.mp4"
    extension = os.path.splitext(original_name)[1].lower()
    if extension not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        return render_template(
            "index.html",
            video_prediction="Unsupported video type. Use MP4, AVI, MOV, MKV, or WebM.",
        )

    stored_name = f"{uuid4().hex}_{original_name}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
    file.save(filepath)
    try:
        details = predict_video(filepath)
        result = details["prediction"]
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        result = str(error)
    finally:
        try:
            os.remove(filepath)
        except OSError:
            pass
    return render_template(
        "index.html", video_prediction=result, video_filename=original_name)


@app.route("/api/predict-video", methods=["POST"])
def predict_video_api():
    if "video" not in request.files or request.files["video"].filename == "":
        return jsonify({"error": "Please select a video."}), 400
    file = request.files["video"]
    original_name = secure_filename(file.filename) or "upload.mp4"
    extension = os.path.splitext(original_name)[1].lower()
    if extension not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        return jsonify({
            "error": "Unsupported video type. Use MP4, AVI, MOV, MKV, or WebM."
        }), 400
    stored_name = f"{uuid4().hex}_{original_name}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
    file.save(filepath)
    try:
        details = predict_video(filepath)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 400
    finally:
        try:
            os.remove(filepath)
        except OSError:
            pass
    return jsonify({"filename": original_name, **details})


@app.route("/predict-news", methods=["POST"])
def predict_news_route():
    text = request.form.get("news_text", "").strip()
    pdf = request.files.get("news_pdf")
    filename = None
    try:
        if pdf and pdf.filename:
            filename = secure_filename(pdf.filename)
            if not filename.lower().endswith(".pdf"):
                raise ValueError("Upload a PDF file with a .pdf extension.")
            text = extract_pdf_text(pdf.stream)
        elif not text:
            raise ValueError("Paste news text or upload a PDF.")
        result = predict_news(text)
    except (ValueError, FileNotFoundError) as error:
        result = str(error)
    return render_template(
        "index.html", news_prediction=result,
        news_text="" if filename else text,
        news_filename=filename,
    )


@app.route("/api/predict-news", methods=["POST"])
def predict_news_api():
    """JSON fake-news endpoint used by the Android application."""
    try:
        pdf = request.files.get("file")
        if pdf and pdf.filename:
            filename = secure_filename(pdf.filename)
            if not filename.lower().endswith(".pdf"):
                raise ValueError("Upload a PDF file with a .pdf extension.")
            text = extract_pdf_text(pdf.stream)
        else:
            payload = request.get_json(silent=True) or {}
            text = str(payload.get("text", "")).strip()
            filename = None
            if not text:
                raise ValueError("News text or a PDF file is required.")
        result = predict_news(text)
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 400
    response = {"prediction": result}
    if filename:
        response["filename"] = filename
        response["extracted_characters"] = len(text)
    return jsonify(response)


@app.route("/api/predict-news/batch", methods=["POST"])
def predict_news_batch_api():
    """Vectorized endpoint for efficiently checking up to 100 articles."""
    payload = request.get_json(silent=True) or {}
    texts = payload.get("texts")
    if not isinstance(texts, list):
        return jsonify({"error": "'texts' must be a JSON array."}), 400
    try:
        predictions = predict_news_batch(texts)
    except (TypeError, ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 400
    return jsonify({"predictions": predictions, "count": len(predictions)})

# Run Flask app
if __name__ == "__main__":
    # Port 5001 avoids stale development-server instances left on port 5000.
    # Disabling the reloader also prevents duplicate TensorFlow models in RAM.
    app.run(host="0.0.0.0", port=5001, debug=False)
