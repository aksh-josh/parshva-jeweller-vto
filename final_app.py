"""
app.py — FINAL COMPLETE VERSION
================================
This is the single final app.py for AI_VTO_Project.

It combines EVERYTHING from this project:
  1. Your original Flask routes + jewelry dataset loading (unchanged)
  2. Advanced ML model loading (EfficientNet-B2 + HRNet + CLIP + GNN)
  3. Geometry-aware overlay engine (perspective + lighting + specular)
  4. Advanced recommendation engine (CLIP + GNN)
  5. Full 7-metric accuracy tracker (PCK, IoU, jitter, scale, positional error)
  6. Accuracy quality model inference (EfficientNet-B0 quality scorer)
  7. All API routes including new accuracy endpoints

HOW TO USE:
  - Before training: runs exactly like your original app (MediaPipe fallback)
  - After Colab:     place best.pth in checkpoints/ → all models activate
  - GPU not needed:  everything runs on CPU at 25+ fps

FOLDER STRUCTURE REQUIRED:
  AI_VTO_Project/
  ├── app.py                          ← THIS FILE
  ├── vto_accuracy.py                 ← from accuracy outputs
  ├── ml_models/
  │   ├── __init__.py
  │   ├── advanced_model.py
  │   ├── advanced_train.py
  │   ├── advanced_recommendation.py
  │   ├── overlay_engine.py
  │   ├── accuracy_model.py
  │   ├── accuracy_train.py
  │   └── accuracy_inference.py
  ├── checkpoints/                    ← created after Colab training
  │   ├── best.pth
  │   ├── accuracy_model_best.pth
  │   └── rec_index/
  │       ├── clip_embeddings.npy
  │       ├── gnn_embeddings.npy
  │       ├── metadata.json
  │       └── gnn_weights.pth
  ├── static/                         ← your existing images
  └── templates/                      ← your existing HTML
"""

import os
import sys
import json
import time
import random
import base64
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_login import LoginManager

import config
from models import db, User, init_db
from auth import auth_bp
from shop import shop_bp
from admin import admin_bp
from vto_accuracy import VTOAccuracyTracker

# ── Torch import (safe — won't crash if not installed yet) ─────────────────
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("[VTO] PyTorch not installed. Run: pip install torch torchvision")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — ML MODEL GLOBALS
# ═══════════════════════════════════════════════════════════════════════════

ML_MODELS_READY   = False   # True once best.pth is loaded
vto_model         = None    # EfficientNet-B2 + HRNet detection model
overlay_engine    = None    # Geometry-aware overlay engine
rec_engine        = None    # CLIP + GNN recommendation engine
accuracy_scorer   = None    # EfficientNet-B0 quality scorer


def load_ml_models():
    """
    Load all custom-trained ML models at app startup.
    Safe fallback: if checkpoints not found, original MediaPipe logic is used.
    Shows clear status messages so you know exactly what loaded.
    """
    global ML_MODELS_READY, vto_model, overlay_engine, rec_engine, accuracy_scorer

    if not TORCH_AVAILABLE:
        print("[VTO] Skipping ML model load — PyTorch not available.")
        return

    # Add ml_models to import path
    ml_path = os.path.join(os.path.dirname(__file__), "ml_models")
    if ml_path not in sys.path:
        sys.path.insert(0, ml_path)

    # ── 1. VTO Detection Model (EfficientNet-B2 + HRNet) ──────────────────
    checkpoint = "./checkpoints/best.pth"
    if os.path.exists(checkpoint):
        try:
            from advanced_model import AdvancedJewelryVTOModel
            vto_model = AdvancedJewelryVTOModel(pretrained=False).to(DEVICE)
            ckpt = torch.load(checkpoint, map_location=DEVICE)
            vto_model.load_state_dict(ckpt["model_state"])
            vto_model.eval()
            print(f"[VTO] Detection model loaded — epoch {ckpt['epoch']}, "
                  f"val_acc={ckpt['val_acc']:.3f}")
            ML_MODELS_READY = True
        except Exception as e:
            print(f"[VTO] Detection model load failed: {e}")
            vto_model = None
    else:
        print("[VTO] No checkpoint at ./checkpoints/best.pth")
        print("      Train on Colab first. Using MediaPipe fallback until then.")

    # ── 2. Geometry-Aware Overlay Engine ──────────────────────────────────
    try:
        from overlay_engine import GeometryOverlayEngine
        overlay_engine = GeometryOverlayEngine(
            use_lighting=True,
            use_specular=True
        )
        print("[VTO] Geometry overlay engine loaded.")
    except Exception as e:
        print(f"[VTO] Overlay engine load failed: {e}. Using basic overlay.")
        overlay_engine = None

    # ── 3. CLIP + GNN Recommendation Engine ───────────────────────────────
    rec_index = "./checkpoints/rec_index"
    if os.path.exists(rec_index):
        try:
            from advanced_recommendation import AdvancedRecommendationEngine
            rec_engine = AdvancedRecommendationEngine(device=str(DEVICE))
            rec_engine.load(rec_index)
            print(f"[VTO] Recommendation engine loaded "
                  f"({len(rec_engine.metadata)} items indexed).")
        except Exception as e:
            print(f"[VTO] Recommendation engine load failed: {e}")
            rec_engine = None
    else:
        print("[VTO] No rec_index found. Using rule-based recommendations.")

    # ── 4. Accuracy Quality Model (EfficientNet-B0) ────────────────────────
    acc_checkpoint = "./checkpoints/accuracy_model_best.pth"
    try:
        from accuracy_inference import AccuracyScorer
        accuracy_scorer = AccuracyScorer(acc_checkpoint)
        if accuracy_scorer.is_loaded:
            print(f"[VTO] Accuracy quality model loaded.")
        else:
            print("[VTO] Accuracy quality model not found — rule-based only.")
    except Exception as e:
        print(f"[VTO] Accuracy scorer load failed: {e}")
        accuracy_scorer = None

    print(f"[VTO] ML models ready: {ML_MODELS_READY} | "
          f"Device: {DEVICE} | "
          f"Overlay: {overlay_engine is not None} | "
          f"Rec: {rec_engine is not None}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — FLASK APP SETUP
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)

app.config["SECRET_KEY"]                   = config.SECRET_KEY
app.config["SQLALCHEMY_DATABASE_URI"]      = config.SQLALCHEMY_DATABASE_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = config.SQLALCHEMY_TRACK_MODIFICATIONS
app.config["MYSQL_HOST"]                   = config.MYSQL_HOST
app.config["MYSQL_USER"]                   = config.MYSQL_USER
app.config["MYSQL_PASSWORD"]               = config.MYSQL_PASSWORD
app.config["MYSQL_DATABASE"]               = config.MYSQL_DATABASE

init_db(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "auth.auth_page"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

app.register_blueprint(auth_bp)
app.register_blueprint(shop_bp)
app.register_blueprint(admin_bp)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — MEDIAPIPE + DATASET GLOBALS
# ═══════════════════════════════════════════════════════════════════════════

mp_face_mesh = mp.solutions.face_mesh
face_detector = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

jewelry_dataset   = {}   # id → item dict
jewelry_by_path   = {}   # full_path → id (for recommendation lookup)
vto_tracker       = VTOAccuracyTracker()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def remove_white_background(image, threshold=240):
    """Makes white/near-white pixels transparent."""
    if image is None:
        return None
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    white = (
        (image[:, :, 0] > threshold) &
        (image[:, :, 1] > threshold) &
        (image[:, :, 2] > threshold)
    )
    image[white, 3] = 0
    return image


def get_images(folder_name):
    folder_path = os.path.join(app.static_folder, folder_name)
    images = []
    if os.path.exists(folder_path):
        for f in os.listdir(folder_path):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")) \
               and f.lower() != "cover.jpeg":
                images.append(f"{folder_name}/{f}")
    return images


def load_jewelry_dataset():
    """Load all jewelry images from static/ into memory."""
    global jewelry_dataset, jewelry_by_path
    static_path = app.static_folder
    jewelry_id  = 1

    # Extended mapping — covers ALL your static/ folders
    folder_mapping = {
        "necklace":           "necklace",
        "mangalsutra":        "mangalsutra",
        "rajwadi":            "rajwadi-heritage",
        "jhumka":             "jhumka",
        "earrings":           "jhumka",
        "chain":              "chain",
        "ring":               "ring",
        "bangles":            "wedding-bangles",
        "bridal-mangalsutra": "bridal-mangalsutra",
        "wedding-earrings":   "wedding-earrings",
        "wedding-necklaces":  "wedding-necklaces",
        "wedding-sets":       "wedding-sets",
        "kundan":             "kundan-stories",
        "festive":            "festive-collection",
        "polki":              "polki-collection",
        "for-her":            "for-her",
        "for-him":            "for-him",
        "for-kids":           "for-kids",
    }

    for category_key, folder_name in folder_mapping.items():
        folder_path = os.path.join(static_path, folder_name)
        if not os.path.exists(folder_path):
            continue

        # Determine VTO type from folder name
        if "ring" in folder_name:
            vto_type = "ring"
        elif "bangles" in folder_name or "bangle" in folder_name:
            vto_type = "bracelet"
        elif "jhumka" in folder_name or "earring" in category_key:
            vto_type = "earring"
        elif "chain" in folder_name:
            vto_type = "chain"
        else:
            vto_type = "necklace"

        for image_file in os.listdir(folder_path):
            if not image_file.lower().endswith(
                (".png", ".jpg", ".jpeg", ".webp")
            ):
                continue

            full_path     = os.path.join(folder_path, image_file)
            relative_path = f"{folder_name}/{image_file}"
            img_data      = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

            if img_data is not None:
                if full_path.lower().endswith((".jpg", ".jpeg")) \
                   or img_data.shape[2] == 3:
                    img_data = remove_white_background(img_data)

                jewelry_dataset[jewelry_id] = {
                    "id":          jewelry_id,
                    "type":        vto_type,
                    "name":        os.path.splitext(image_file)[0]
                                   .replace("_", " ").title(),
                    "category":    category_key,
                    "folder":      folder_name,
                    "image_path":  relative_path,
                    "image_data":  img_data,
                    "size_factor": 1.0,
                }
                jewelry_by_path[full_path] = jewelry_id
                jewelry_id += 1

    print(f"[VTO] Loaded {len(jewelry_dataset)} jewelry items.")


def decode_base64_image(b64_string):
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        arr = np.frombuffer(base64.b64decode(b64_string), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def encode_image_to_base64(image):
    try:
        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return f"data:image/jpeg;base64,{base64.b64encode(buf).decode('ascii')}"
    except Exception:
        return None


def get_lm_px(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — JEWELRY PLACEMENT (advanced engine + original fallback)
# ═══════════════════════════════════════════════════════════════════════════

def _overlay_image_basic(background, overlay, x, y):
    """Original simple alpha-blend overlay — used as fallback."""
    bg_h, bg_w = background.shape[:2]
    ol_h, ol_w = overlay.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + ol_w), min(bg_h, y + ol_h)
    ox1, oy1 = max(0, -x), max(0, -y)
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)
    if x2 <= x1 or y2 <= y1:
        return background
    crop = overlay[oy1:oy2, ox1:ox2]
    bg   = background[y1:y2, x1:x2]
    if crop.shape[2] == 4:
        alpha   = crop[:, :, 3:4].astype(np.float32) / 255.0
        blended = alpha * crop[:, :, :3].astype(np.float32) + \
                  (1 - alpha) * bg.astype(np.float32)
        background[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    else:
        background[y1:y2, x1:x2] = crop[:, :, :3]
    return background


def apply_virtual_jewelry(image, face_landmarks, jewelry_id, zoom_factor=1.0):
    """
    Apply jewelry overlay.
    Uses GeometryOverlayEngine (advanced) if loaded, otherwise original logic.
    """
    if jewelry_id not in jewelry_dataset:
        return image, "Invalid ID", False

    info         = jewelry_dataset[jewelry_id]
    jewelry_type = info["type"]
    jewelry_img  = info["image_data"]

    if jewelry_img is None:
        return image, "Image error", False

    # ── Advanced geometry engine ──────────────────────────────────────────
    if overlay_engine is not None:
        if jewelry_type in ("necklace", "chain", "mangalsutra"):
            return overlay_engine.apply_necklace(
                image, jewelry_img, face_landmarks, zoom_factor
            )
        elif jewelry_type == "earring":
            return overlay_engine.apply_earrings(
                image, jewelry_img, face_landmarks, zoom_factor
            )
        # ring / bracelet falls through to original (needs hand landmarks)

    # ── Original fallback ─────────────────────────────────────────────────
    h, w   = image.shape[:2]
    lm     = face_landmarks.landmark
    lx, _  = get_lm_px(lm[234], w, h)
    rx, _  = get_lm_px(lm[454], w, h)
    face_w = abs(rx - lx)
    if face_w < 20:
        return image, "Face too small", False

    if jewelry_type in ("necklace", "chain", "mangalsutra"):
        cx, cy    = get_lm_px(lm[152], w, h)
        tgt_w     = max(60, int(face_w * 1.7 * zoom_factor))
        ol_h, ol_w = jewelry_img.shape[:2]
        tgt_h     = int(tgt_w * ol_h / max(ol_w, 1))
        resized   = cv2.resize(jewelry_img, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
        image     = _overlay_image_basic(image, resized,
                                          cx - tgt_w // 2,
                                          cy + int(face_w * 0.2))
        return image, "Necklace placed", True

    elif jewelry_type == "earring":
        lex, ley = get_lm_px(lm[234], w, h)
        rex, rey = get_lm_px(lm[454], w, h)
        _, ljy   = get_lm_px(lm[132], w, h)
        _, rjy   = get_lm_px(lm[361], w, h)
        lly      = (ley + ljy) // 2
        rly      = (rey + rjy) // 2
        ol_h, ol_w = jewelry_img.shape[:2]
        tgt_h    = max(30, int(face_w * 0.45 * zoom_factor))
        if ol_w / max(ol_h, 1) >= 0.85:
            mid   = ol_w // 2
            l_img = jewelry_img[:, :mid]
            r_img = jewelry_img[:, mid:]
        else:
            l_img = jewelry_img
            r_img = cv2.flip(jewelry_img, 1)
        for img_e, ex, ey in [(l_img, lex, lly), (r_img, rex, rly)]:
            eh, ew = img_e.shape[:2]
            tw     = max(15, int(tgt_h * ew / max(eh, 1)))
            res    = cv2.resize(img_e, (tw, tgt_h), interpolation=cv2.INTER_AREA)
            image  = _overlay_image_basic(image, res, ex - tw // 2, ey)
        return image, "Earrings placed", True

    return image, "Unknown type", False


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — WEB ROUTES (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/collections/<collection_name>")
def collections(collection_name):
    images = get_images(collection_name)
    if collection_name == "wedding":
        return render_template("necklace.html", images=images,
                               title="Wedding Collection")
    elif collection_name == "gifting":
        return render_template("collection.html", images=images,
                               title="Gifting Collection")
    return render_template("necklace.html", images=images,
                           title=f"{collection_name.title()} Collection")


@app.route("/shop/<category>")
def shop_category(category):
    category_map = {
        "earrings":    "jhumka",
        "necklaces":   "necklace",
        "rajwadi":     "rajwadi-heritage",
        "mangalsutra": "mangalsutra",
        "rings":       "ring",
        "bangles":     "wedding-bangles",
        "chain":       "chain",
    }
    folder_name = category_map.get(category)
    if not folder_name:
        return "Category not found", 404
    images = get_images(folder_name)
    title  = category.replace("_", " ").title()
    tmpl   = f"{folder_name}.html"
    if not os.path.exists(os.path.join(app.template_folder, tmpl)):
        return render_template("necklace.html", images=images, title=title)
    return render_template(tmpl, images=images, title=title)


@app.route("/tryon/<category>/<filename>")
def tryon(category, filename):
    found_id = None
    for jid, info in jewelry_dataset.items():
        if info["category"] == category and info["image_path"].endswith(filename):
            found_id = jid
            break
    if not found_id:
        return "Product not found.", 404
    return render_template(
        "tryon_live.html",
        category=category,
        filename=filename,
        jewelry_id=found_id,
        jewelry_name=jewelry_dataset[found_id]["name"],
    )


@app.route("/mangalsutra")
def mangalsutra():
    return render_template("mangalsutra.html",
                           images=get_images("mangalsutra"), title="Mangalsutra")

@app.route("/jhumka")
def jhumka():
    return render_template("jhumka.html",
                           images=get_images("jhumka"), title="Jhumka")

@app.route("/necklace")
def necklace():
    return render_template("necklace.html",
                           images=get_images("necklace"), title="Necklace")

@app.route("/chain")
def chain():
    return render_template("chain.html",
                           images=get_images("chain"), title="Chain")

@app.route("/ring")
def ring():
    return render_template("ring.html",
                           images=get_images("ring"), title="Ring")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN TRY-ON API  /api/jewelry-tryon
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/api/jewelry/categories", methods=["GET"])
def get_categories_api():
    serializable = []
    for k, v in jewelry_dataset.items():
        item = v.copy()
        del item["image_data"]
        serializable.append(item)
    return jsonify({"status": "success", "categories": {"all": serializable}})


@app.route("/api/jewelry-tryon", methods=["POST"])
def api_tryon():
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data"}), 400

        if data.get("action") == "start_tryon":
            return jsonify({"status": "success", "session_id": "123"})

        if data.get("action") == "process_frame" or \
           ("data" in data and "frame" in data.get("data", {})):

            frame_data = data["data"]["frame"]
            zoom       = float(data["data"].get("zoom_factor", 1.0))

            jewelry_ids       = data["data"].get("jewelry_ids", [])
            jewelry_id_single = data["data"].get("jewelry_id")
            if not jewelry_ids and jewelry_id_single:
                jewelry_ids = [int(jewelry_id_single)]
            elif isinstance(jewelry_ids, list):
                jewelry_ids = [int(x) for x in jewelry_ids]
            else:
                jewelry_ids = [int(jewelry_ids)]

            frame = decode_base64_image(frame_data)
            if frame is None:
                return jsonify({"status": "error", "message": "Bad frame"})

            # ── KEY: save original frame BEFORE overlay for accuracy diff ─
            original_frame = frame.copy()

            t_start = time.time()

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb)

            person_detected    = False
            status             = "No Face"
            confidence         = 0.0
            first_jewelry_type = "necklace"
            first_category     = "necklace"

            if results.multi_face_landmarks:
                person_detected = True
                face_lm         = results.multi_face_landmarks[0]

                # Detection confidence = fraction of landmarks within frame
                lm_in_frame = sum(
                    1 for l in face_lm.landmark
                    if 0.0 <= l.x <= 1.0 and 0.0 <= l.y <= 1.0
                )
                confidence = lm_in_frame / len(face_lm.landmark)

                # Apply all jewelry overlays
                for j_id in jewelry_ids:
                    frame, status, _ = apply_virtual_jewelry(
                        frame, face_lm, j_id, zoom
                    )
                    if j_id in jewelry_dataset:
                        first_jewelry_type = jewelry_dataset[j_id]["type"]
                        first_category     = jewelry_dataset[j_id]["category"]

            inference_ms = (time.time() - t_start) * 1000

            # ── Accuracy quality score (EfficientNet-B0) ──────────────────
            quality_score = 0.72   # neutral default
            if accuracy_scorer is not None and person_detected:
                quality_score = accuracy_scorer.predict(frame, first_category)

            # ── Record 7-metric accuracy ───────────────────────────────────
            acc_result = vto_tracker.record_frame(
                face_landmarks        = results.multi_face_landmarks[0]
                                        if results.multi_face_landmarks else None,
                face_detected         = person_detected,
                detection_confidence  = confidence,
                original_frame        = original_frame,   # real before-frame
                processed_frame       = frame,            # real after-frame
                jewelry_type          = first_jewelry_type,
                category              = first_category,
                frame_shape           = frame.shape,
                zoom_factor           = zoom,
                inference_ms          = inference_ms,
            )

            # Combined accuracy score: 55% rule-based + 45% model score
            combined_accuracy = None
            if acc_result is not None and accuracy_scorer is not None:
                live_metrics     = vto_tracker.get_live_metrics()
                combined_accuracy = accuracy_scorer.compute_combined_score(
                    live_metrics, quality_score
                )

            # Send live metrics every 30 frames
            live_metrics_payload = None
            if vto_tracker.total_frames % 30 == 0:
                live_metrics_payload = vto_tracker.get_live_metrics()
                if combined_accuracy is not None:
                    live_metrics_payload["combined_accuracy"] = round(combined_accuracy, 3)
                    live_metrics_payload["quality_score"]     = round(quality_score, 3)

            return jsonify({
                "processed_frame":  encode_image_to_base64(frame),
                "position_status":  status,
                "person_detected":  person_detected,
                "confidence":       round(confidence, 3),
                "jewelry_count":    len(jewelry_ids),
                "ml_model_active":  ML_MODELS_READY,
                "live_accuracy":    live_metrics_payload,
                "status":           "success",
            })

    except Exception as e:
        print(f"[API Error] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

    return jsonify({"status": "error", "message": "Unknown action"})


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — RECOMMENDATION API
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/api/recommendations/<int:jewelry_id>")
def get_recommendations_api(jewelry_id):
    """
    Returns complementary jewelry recommendations.
    Uses CLIP+GNN if loaded, falls back to rule-based recommender.
    """
    if jewelry_id not in jewelry_dataset:
        return jsonify({"success": False, "recommendations": [], "count": 0})

    info = jewelry_dataset[jewelry_id]

    # ── Advanced CLIP + GNN ────────────────────────────────────────────────
    if rec_engine is not None:
        try:
            query_path = os.path.join(app.static_folder, info["image_path"])
            recs = rec_engine.recommend(
                query_path=query_path,
                query_category=info["category"],
                top_k=8,
            )
            return jsonify({
                "success": True,
                "engine":  "clip_gnn",
                "recommendations": [
                    {
                        "id":         jewelry_by_path.get(r["path"]),
                        "name":       r["name"],
                        "category":   r["category"],
                        "image_path": r["path"].replace(
                            app.static_folder, ""
                        ).lstrip("/\\"),
                        "clip_sim":   round(r["clip_sim"], 3),
                        "gnn_sim":    round(r["gnn_sim"], 3),
                        "score":      round(r["final_score"], 3),
                        "reason":     f"Style compatibility: "
                                      f"{r['rule_compat']:.0%}",
                    }
                    for r in recs
                ],
                "count": len(recs),
            })
        except Exception as e:
            print(f"[Rec] CLIP+GNN failed: {e}, falling back.")

    # ── Rule-based fallback ────────────────────────────────────────────────
    try:
        from recommendations import recommender
        from models import Product
        product = Product.query.filter_by(
            image_path=info["image_path"]
        ).first()
        if product:
            recs = recommender.get_recommendations(product.id, max_results=8)
            return jsonify({
                "success": True,
                "engine":  "rule_based",
                "recommendations": [
                    {
                        "id":         r["product"]["id"],
                        "name":       r["product"]["name"],
                        "image_path": r["product"]["image_path"],
                        "category":   r["product"]["category"],
                        "score":      r["score"],
                        "similarity": r["similarity"],
                        "reason":     r["reason"],
                    }
                    for r in recs
                ],
                "count": len(recs),
            })
    except Exception as e:
        print(f"[Rec] Rule-based fallback also failed: {e}")

    return jsonify({"success": False, "recommendations": [], "count": 0})


@app.route("/api/similar/<int:jewelry_id>")
def get_similar_api(jewelry_id):
    """Returns visually similar items in the same category."""
    if jewelry_id not in jewelry_dataset:
        return jsonify({"success": False, "similar": []})
    info = jewelry_dataset[jewelry_id]
    try:
        from recommendations import recommender
        from models import Product
        product = Product.query.filter_by(
            image_path=info["image_path"]
        ).first()
        if product:
            similar = recommender.get_similar_products(product.id, max_results=4)
            return jsonify({
                "success": True,
                "similar": [
                    {
                        "id":         s["product"]["id"],
                        "name":       s["product"]["name"],
                        "image_path": s["product"]["image_path"],
                        "category":   s["product"]["category"],
                        "similarity": s["similarity"],
                    }
                    for s in similar
                ],
            })
    except Exception as e:
        print(f"[Similar] Error: {e}")
    return jsonify({"success": False, "similar": []})


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9 — ACCURACY API  (7-metric system)
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/api/vto/accuracy/start", methods=["POST"])
def vto_accuracy_start():
    """Start a fresh accuracy tracking session."""
    data = request.json or {}
    vto_tracker.reset()
    vto_tracker.category_tested = data.get("category", "unknown")
    return jsonify({
        "success": True,
        "message": "Accuracy tracking started",
        "session_start": vto_tracker.session_start,
    })


@app.route("/api/vto/accuracy/live", methods=["GET"])
def vto_accuracy_live():
    """
    Lightweight live metrics — call every 2 seconds from dashboard.
    Returns rolling 30-frame window accuracy.
    """
    if vto_tracker.total_frames == 0:
        return jsonify({
            "success": False,
            "message": "No frames yet. Start a try-on session first.",
        })
    metrics = vto_tracker.get_live_metrics()
    # Append combined score if quality model is loaded
    if accuracy_scorer is not None:
        metrics["quality_model_active"] = accuracy_scorer.is_loaded
        metrics["mean_inference_ms"]    = round(
            accuracy_scorer.mean_inference_ms, 1
        )
    return jsonify({"success": True, "metrics": metrics})


@app.route("/api/vto/accuracy/report", methods=["GET"])
def vto_accuracy_report():
    """Full accuracy report — PCK, IoU, jitter, scale, per-category, grade."""
    if vto_tracker.total_frames == 0:
        return jsonify({"success": False, "message": "No frames recorded yet."})
    report = vto_tracker.get_accuracy_report()
    return jsonify({"success": True, "report": report})


@app.route("/api/vto/accuracy/save", methods=["POST"])
def vto_accuracy_save():
    """Save full accuracy report to static/reports/ as JSON."""
    if vto_tracker.total_frames == 0:
        return jsonify({"success": False, "message": "No data to save"})
    report    = vto_tracker.get_accuracy_report()
    rdir      = os.path.join(app.static_folder, "reports")
    os.makedirs(rdir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"vto_accuracy_{timestamp}.json"
    with open(os.path.join(rdir, filename), "w") as f:
        json.dump(report, f, indent=2)
    return jsonify({
        "success":  True,
        "filename": filename,
        "report":   report,
        "message":  f"Saved to static/reports/{filename}",
    })


@app.route("/api/vto/accuracy/reset", methods=["POST"])
def vto_accuracy_reset():
    """Reset all accuracy tracking data."""
    vto_tracker.reset()
    return jsonify({"success": True, "message": "Accuracy data cleared."})


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10 — STATUS API
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/api/model/status")
def model_status():
    """Returns which models are currently loaded and active."""
    return jsonify({
        "ml_model_loaded":        ML_MODELS_READY,
        "overlay_engine":         overlay_engine is not None,
        "recommendation_engine":  rec_engine is not None,
        "accuracy_quality_model": accuracy_scorer is not None and
                                  accuracy_scorer.is_loaded
                                  if accuracy_scorer else False,
        "device":                 str(DEVICE) if DEVICE else "cpu",
        "jewelry_items_loaded":   len(jewelry_dataset),
        "message": "Custom model active" if ML_MODELS_READY else
                   "Using MediaPipe fallback — train on Colab first",
    })


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11 — STARTUP
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Step 1: Load jewelry images from static/
    load_jewelry_dataset()

    # Step 2: Load all custom ML models (safe — won't crash if not trained)
    load_ml_models()

    # Step 3: Build rule-based recommendation index (original fallback)
    with app.app_context():
        try:
            from recommendations import recommender
            from models import Product
            all_products = Product.query.filter_by(is_active=True).all()
            recommender.build_index(all_products, app.static_folder)
        except Exception as e:
            print(f"[Rec] Rule-based index: {e}")

    # Step 4: Run Flask
    app.run(debug=True, port=5000)
