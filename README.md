# Object Lens

A small Flask web interface for the trained object classifier in `saved_model/`.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000` and upload a JPG, PNG, or WebP image (up to 10 MB).

The first classification may take longer because torchvision may need to download ResNet-50 weights. The model files in `saved_model/` must remain in place.

## Command line

```bash
python3 detect.py path/to/image.jpg
```

Optional controls are available with `--threshold` and `--min-votes`.

The optional notebook dependencies are in `requirements-notebook.txt`. Its
legacy ReliefF analysis requires Python 3.10 or earlier because `skrebate` is
not compatible with Python 3.12.
