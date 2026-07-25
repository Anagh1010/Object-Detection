"""Flask application for uploading and classifying an image."""

from __future__ import annotations

import base64
from io import BytesIO

from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

from detect import classify_image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.errorhandler(413)
def file_too_large(_error):
    return render_template("index.html", error="Please upload an image smaller than 10 MB."), 413


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    detections = None
    image_data = None

    if request.method == "POST":
        uploaded = request.files.get("image")
        if not uploaded or not uploaded.filename:
            error = "Choose an image to classify."
        elif not allowed_file(uploaded.filename):
            error = "Use a JPG, PNG, or WebP image."
        else:
            try:
                raw_image = uploaded.read()
                with Image.open(BytesIO(raw_image)) as source:
                    image = source.convert("RGB")
                detections = classify_image(image)
                image_data = raw_image
            except UnidentifiedImageError:
                error = "That file is not a readable image."
            except Exception:
                app.logger.exception("Classification failed")
                error = "Classification could not be completed. Check that model files are available."

    return render_template(
        "index.html",
        detections=detections,
        error=error,
        image_data=(base64.b64encode(image_data).decode("ascii") if image_data else None),
    )


if __name__ == "__main__":
    app.run(debug=True)
