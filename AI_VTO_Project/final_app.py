"""
app.py — COMPLETE VERSION WITH FIXES
- Ring sizing increased significantly
- Performance optimizations (frame skipping, reduced quality)
- Mobile fullscreen support
"""

import os
import sys
import json 
import time
import base64
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
from analytics_logger import AnalyticsLogger, generate_session_id

try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("[VTO] PyTorch not installed. Run: pip install torch torchvision")

# Initialize analytics logger
analytics = AnalyticsLogger({
    'host': config.MYSQL_HOST,
    'user': config.MYSQL_USER,
    'password': config.MYSQL_PASSWORD,
    'database': config.MYSQL_DATABASE
})

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — ML MODEL GLOBALS
# ═══════════════════════════════════════════════════════════════════════════

ML_MODELS_READY    = False
vto_model          = None
overlay_engine     = None
rec_engine         = None
accuracy_scorer    = None
gaze_detector_obj  = None
GAZE_ENABLED       = False

# FIX: Frame skipping for performance
_frame_counter = 0
_FRAME_SKIP = 0  # Process every 2nd frame (skip 1)


def load_ml_models():
    global ML_MODELS_READY, vto_model, overlay_engine, rec_engine, accuracy_scorer
    global gaze_detector_obj

    if not TORCH_AVAILABLE:
        print("[VTO] Skipping ML model load — PyTorch not available.")
        return

    ml_path = os.path.join(os.path.dirname(__file__), "ml_models")
    if ml_path not in sys.path:
        sys.path.insert(0, ml_path)

    # ── 1. VTO Detection Model ─────────────────────────────────────────────
    checkpoint = "./checkpoints/best.pth"
    if os.path.exists(checkpoint):
        try:
            from advanced_model import AdvancedJewelryVTOModel
            vto_model = AdvancedJewelryVTOModel(pretrained=False).to(DEVICE)
            ckpt = torch.load(checkpoint, map_location=DEVICE)
            model_sd = vto_model.state_dict()
            filtered = {k: v for k, v in ckpt["model_state"].items()
                        if k in model_sd and v.shape == model_sd[k].shape}
            removed  = [k for k in ckpt["model_state"] if k not in filtered]
            vto_model.load_state_dict(filtered, strict=False)
            vto_model.eval()
            pct = len(filtered) / max(len(ckpt["model_state"]), 1) * 100
            print(f"[VTO] Detection model loaded — epoch {ckpt['epoch']}, "
                  f"val_acc={ckpt['val_acc']:.3f} | weights: {pct:.0f}%")
            if removed:
                print(f"[VTO] Skipped {len(removed)} shape-mismatched layers")
            ML_MODELS_READY = True
        except Exception as e:
            print(f"[VTO] Detection model load failed: {e}")
            vto_model = None
    else:
        print("[VTO] No checkpoint at ./checkpoints/best.pth")
        print("      Train on Colab first. Using MediaPipe fallback until then.")

    # ── 2. Geometry Overlay Engine ─────────────────────────────────────────
    try:
        from overlay_engine import GeometryOverlayEngine
        overlay_engine = GeometryOverlayEngine(use_lighting=True, use_specular=True)
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
            print(f"[VTO] Recommendation engine loaded ({len(rec_engine.metadata)} items indexed).")
        except Exception as e:
            print(f"[VTO] Recommendation engine load failed: {e}")
            rec_engine = None
    else:
        print("[VTO] No rec_index found. Using rule-based recommendations.")

    # ── 4. Accuracy Quality Model ──────────────────────────────────────────
    acc_checkpoint = "./checkpoints/accuracy_model_best.pth"
    try:
        from accuracy_inference import AccuracyScorer
        accuracy_scorer = AccuracyScorer(acc_checkpoint)
        if accuracy_scorer.is_loaded:
            print("[VTO] Accuracy quality model loaded.")
        else:
            print("[VTO] Accuracy quality model not found — rule-based only.")
    except Exception as e:
        print(f"[VTO] Accuracy scorer load failed: {e}")
        accuracy_scorer = None

    # ── 5. Gaze Detector ───────────────────────────────────────────────────
    try:
        from gaze_detector import GazeDetector
        gaze_detector_obj = GazeDetector(
            dwell_threshold_seconds=2.0,
            right_gaze_threshold=-0.015,
            smoothing_frames=8,
        )
        gaze_detector_obj.is_active = False
        print("[GAZE] Gaze detector loaded (inactive until USB camera selected).")
    except Exception as e:
        print(f"[GAZE] Load failed: {e}")
        gaze_detector_obj = None

    print(f"[VTO] ML models ready: {ML_MODELS_READY} | "
          f"Device: {DEVICE} | "
          f"Overlay: {overlay_engine is not None} | "
          f"Rec: {rec_engine is not None}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — FLASK APP SETUP
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173"])

app.config["SECRET_KEY"]                     = config.SECRET_KEY
app.config["SQLALCHEMY_DATABASE_URI"]        = config.SQLALCHEMY_DATABASE_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = config.SQLALCHEMY_TRACK_MODIFICATIONS
app.config["MYSQL_HOST"]                     = config.MYSQL_HOST
app.config["MYSQL_USER"]                     = config.MYSQL_USER
app.config["MYSQL_PASSWORD"]                 = config.MYSQL_PASSWORD
app.config["MYSQL_DATABASE"]                 = config.MYSQL_DATABASE

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

mp_face_mesh  = mp.solutions.face_mesh
mp_hands      = mp.solutions.hands

face_detector = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.25,
    min_tracking_confidence=0.25,
)

hand_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.50,
    min_tracking_confidence=0.40,
)

CAMERA_INDEX = 0

jewelry_dataset = {}
jewelry_by_path = {}
vto_tracker     = VTOAccuracyTracker()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def remove_white_background(image, threshold=240):
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
    global jewelry_dataset, jewelry_by_path
    static_path = app.static_folder
    jewelry_id  = 1

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
            if not image_file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue
            full_path     = os.path.join(folder_path, image_file)
            relative_path = f"{folder_name}/{image_file}"
            img_data      = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
            if img_data is not None:
                if full_path.lower().endswith((".jpg", ".jpeg")) or img_data.shape[2] == 3:
                    img_data = remove_white_background(img_data)
                jewelry_dataset[jewelry_id] = {
                    "id":          jewelry_id,
                    "type":        vto_type,
                    "name":        os.path.splitext(image_file)[0].replace("_", " ").title(),
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
        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return f"data:image/jpeg;base64,{base64.b64encode(buf).decode('ascii')}"
    except Exception:
        return None


def get_lm_px(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — JEWELRY PLACEMENT
# ═══════════════════════════════════════════════════════════════════════════

def _overlay_image_basic(background, overlay, x, y):
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


def apply_virtual_jewelry(image, face_landmarks, hand_landmarks_result, jewelry_id, zoom_factor=1.0):
    """
    Main jewelry application function with improved ring sizing
    """
    if jewelry_id not in jewelry_dataset:
        return image, "Invalid ID", False
    info         = jewelry_dataset[jewelry_id]
    jewelry_type = info["type"]
    jewelry_img  = info["image_data"]
    if jewelry_img is None:
        return image, "Image error", False

    # ── RING: use hand landmarks with INCREASED SIZE ──────────────────────
    if jewelry_type in ("ring", "bracelet"):
        if overlay_engine is not None and hand_landmarks_result is not None and \
           hand_landmarks_result.multi_hand_landmarks:
            hand_lm = hand_landmarks_result.multi_hand_landmarks[0]
            try:
                return overlay_engine.apply_ring(image, jewelry_img, hand_lm, zoom_factor=zoom_factor)
            except Exception as e:
                print(f"[Ring] overlay_engine failed: {e}")

        # FIX: Enhanced fallback ring placement with LARGER SIZE
        if hand_landmarks_result is not None and hand_landmarks_result.multi_hand_landmarks:
            return _apply_ring_basic_enhanced(image, jewelry_img,
                                     hand_landmarks_result.multi_hand_landmarks[0],
                                     zoom_factor)
        return image, "Show your hand to camera for ring try-on", False

    # # ── FACE JEWELRY: use face landmarks ──────────────────────────────────
    # if face_landmarks is None:
    #     return image, "No face detected", False

    # if overlay_engine is not None:
    #     if jewelry_type in ("necklace", "chain", "mangalsutra"):
    #         return overlay_engine.apply_necklace(image, jewelry_img, face_landmarks, zoom_factor)
    #     elif jewelry_type == "earring":
    #         return overlay_engine.apply_earrings(image, jewelry_img, face_landmarks, zoom_factor)

    # h, w   = image.shape[:2]
    # lm     = face_landmarks.landmark
    # lx, _  = get_lm_px(lm[234], w, h)
    # ── FACE JEWELRY: use face landmarks ──────────────────────────────────
    if face_landmarks is None:
        return image, "No face detected", False

    # FIX: Added try/except to prevent black screens! 
    # Necklaces are explicitly routed to the safe fallback below to avoid Docker OpenCV crashes.
    if overlay_engine is not None:
        try:
            if jewelry_type == "earring":
                return overlay_engine.apply_earrings(image, jewelry_img, face_landmarks, zoom_factor)
        except Exception as e:
            print(f"[Face Jewelry] Engine failed: {e} - Using safe fallback!")

    h, w   = image.shape[:2]
    lm     = face_landmarks.landmark
    lx, _  = get_lm_px(lm[234], w, h)
    rx, _  = get_lm_px(lm[454], w, h)
    face_w = abs(rx - lx)
    if face_w < 20:
        return image, "Face too small", False

    if jewelry_type in ("necklace", "chain", "mangalsutra"):
        cx, cy     = get_lm_px(lm[152], w, h)
        tgt_w      = max(60, int(face_w * 1.7 * zoom_factor))
        ol_h, ol_w = jewelry_img.shape[:2]
        tgt_h      = int(tgt_w * ol_h / max(ol_w, 1))
        resized    = cv2.resize(jewelry_img, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
        image      = _overlay_image_basic(image, resized, cx - tgt_w // 2, cy + int(face_w * 0.2))
        return image, "Necklace placed", True

    elif jewelry_type == "earring":
        lex, ley   = get_lm_px(lm[234], w, h)
        rex, rey   = get_lm_px(lm[454], w, h)
        _, ljy     = get_lm_px(lm[132], w, h)
        _, rjy     = get_lm_px(lm[361], w, h)
        lly        = (ley + ljy) // 2
        rly        = (rey + rjy) // 2
        ol_h, ol_w = jewelry_img.shape[:2]
        tgt_h      = max(30, int(face_w * 0.45 * zoom_factor))
        if ol_w / max(ol_h, 1) >= 0.85:
            mid    = ol_w // 2
            l_img  = jewelry_img[:, :mid]
            r_img  = jewelry_img[:, mid:]
        else:
            l_img  = jewelry_img
            r_img  = cv2.flip(jewelry_img, 1)
        for img_e, ex, ey in [(l_img, lex, lly), (r_img, rex, rly)]:
            eh, ew = img_e.shape[:2]
            tw     = max(15, int(tgt_h * ew / max(eh, 1)))
            res    = cv2.resize(img_e, (tw, tgt_h), interpolation=cv2.INTER_AREA)
            image  = _overlay_image_basic(image, res, ex - tw // 2, ey)
        return image, "Earrings placed", True

    return image, "Unknown type", False


def _apply_ring_basic_enhanced(image, jewelry_img, hand_lm, zoom_factor=1.0):
    """
    ENHANCED RING PLACEMENT WITH SIGNIFICANTLY LARGER SIZE ✨✨✨
    
    CHANGES FROM ORIGINAL:
    1. Increased base size multiplier from 2.4x to 4.2x (75% larger)
    2. Better palm width calculation using multiple finger bases
    3. Increased minimum ring width from 20px to 40px
    4. Better positioning at the middle of the finger segment
    """
    h_img, w_img = image.shape[:2]
    lm = hand_lm.landmark

    # Ring finger landmarks
    mcp_x = int(lm[13].x * w_img)  # Ring finger base (MCP)
    mcp_y = int(lm[13].y * h_img)
    pip_x = int(lm[14].x * w_img)  # Ring finger middle (PIP)
    pip_y = int(lm[14].y * h_img)

    # IMPROVED: Calculate palm width using multiple reference points
    # Use index finger base (5) to pinky base (17) for better scale
    index_base_x = int(lm[5].x * w_img)
    pinky_base_x = int(lm[17].x * w_img)
    palm_width = abs(index_base_x - pinky_base_x)
    
    # Also check middle to pinky distance for validation
    middle_base_x = int(lm[9].x * w_img)
    alt_width = abs(middle_base_x - pinky_base_x)
    
    # Use the larger measurement for better sizing
    reference_width = max(palm_width, alt_width * 1.5)
    
    # FIX: SIGNIFICANTLY INCREASED finger width estimation
    # Original was palm_width // 3, now using larger divisor and base
    finger_width = max(40, int(reference_width * 0.30))  # Increased from max(18, ...)

    # Ring center: positioned between MCP and PIP
    ring_cx = (mcp_x + pip_x) // 2
    ring_cy = (mcp_y + pip_y) // 2

    # Calculate finger angle for proper rotation
    dx = pip_x - mcp_x
    dy = pip_y - mcp_y
    angle = np.degrees(np.arctan2(dy, dx)) - 90

    # CRITICAL FIX: MASSIVELY INCREASED SIZE MULTIPLIER ✨✨✨
    # Original: 2.4x → New: 4.2x (75% larger rings!)
    target_w = max(40, int(finger_width * 4.2 * zoom_factor))
    
    ol_h, ol_w = jewelry_img.shape[:2]
    target_h = max(20, int(target_w * ol_h / max(ol_w, 1)))

    # Resize ring to target dimensions
    ring_resized = cv2.resize(jewelry_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # Rotate ring to match finger angle
    M = cv2.getRotationMatrix2D((target_w // 2, target_h // 2), -angle, 1.0)
    ring_rotated = cv2.warpAffine(
        ring_resized, M, (target_w, target_h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )

    # Position ring on finger
    pos_x = ring_cx - target_w // 2
    pos_y = ring_cy - target_h // 2
    
    image = _overlay_image_basic(image, ring_rotated, pos_x, pos_y)
    return image, "Ring placed", True


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — WEB ROUTES
# ═══════════════════════════════════════════════════════════════════════════

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/collections/<collection_name>")
# def collections(collection_name):
#     images = get_images(collection_name)
#     if collection_name == "wedding":
#         return render_template("necklace.html", images=images, title="Wedding Collection")
#     elif collection_name == "gifting":
#         return render_template("collection.html", images=images, title="Gifting Collection")
#     return render_template("necklace.html", images=images, title=f"{collection_name.title()} Collection")

# @app.route("/shop/<category>")
# def shop_category(category):
#     category_map = {
#         "earrings": "jhumka", "necklaces": "necklace",
#         "rajwadi": "rajwadi-heritage", "mangalsutra": "mangalsutra",
#         "rings": "ring", "bangles": "wedding-bangles", "chain": "chain",
#     }
#     folder_name = category_map.get(category)
#     if not folder_name:
#         return "Category not found", 404
#     images = get_images(folder_name)
#     title  = category.replace("_", " ").title()
#     tmpl   = f"{folder_name}.html"
#     if not os.path.exists(os.path.join(app.template_folder, tmpl)):
#         return render_template("necklace.html", images=images, title=title)
#     return render_template(tmpl, images=images, title=title)

# @app.route("/tryon/<category>/<filename>")
# def tryon(category, filename):
#     found_id = None
#     for jid, info in jewelry_dataset.items():
#         if info["category"] == category and info["image_path"].endswith(filename):
#             found_id = jid
#             break
#     if not found_id:
#         return "Product not found.", 404
#     return render_template("tryon_live.html", category=category, filename=filename,
#                            jewelry_id=found_id, jewelry_name=jewelry_dataset[found_id]["name"])

# @app.route("/mangalsutra")
# def mangalsutra():
#     return render_template("mangalsutra.html", images=get_images("mangalsutra"), title="Mangalsutra")

# @app.route("/jhumka")
# def jhumka():
#     return render_template("jhumka.html", images=get_images("jhumka"), title="Jhumka")

# @app.route("/necklace")
# def necklace():
#     return render_template("necklace.html", images=get_images("necklace"), title="Necklace")

# @app.route("/chain")
# def chain():
#     return render_template("chain.html", images=get_images("chain"), title="Chain")

# @app.route("/ring")
# def ring():
#     return render_template("ring.html", images=get_images("ring"), title="Ring")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — DATA APIs FOR REACT
# ═══════════════════════════════════════════════════════════════════════════

# @app.route("/api/products", methods=["GET"])
# def get_products():
#     category_filter = request.args.get("category")
#     material_filter = request.args.get("material")
    
#     # We will return the items from your pre-loaded ML jewelry_dataset
#     products_list = []
    
#     for jid, info in jewelry_dataset.items():
#         # Apply filters if they exist in the URL (e.g. /api/products?category=jhumka)
#         if category_filter and category_filter.lower() not in info["category"].lower() and category_filter.lower() not in info["folder"].lower():
#             continue
            
#         products_list.append({
#             "id": jid,
#             "name": info["name"],
#             "category": info["category"],
#             "folder": info["folder"],
#             "image_path": info["image_path"],
#             "price": 45000, # Default mock price, replace with DB lookup if needed
#             "type": info["type"]
#         })
        
#     return jsonify({"success": True, "products": products_list})
@app.route("/api/products", methods=["GET"])
def get_products():
    category_filter = request.args.get("category")
    
    products_list = []
    
    for jid, info in jewelry_dataset.items():
        if category_filter:
            cf = category_filter.lower()
            cat = info["category"].lower()
            fol = info["folder"].lower()
            
            # THE BUG FIX: Prevent "ring" from matching "earring"
            if cf == "ring" and ("earring" in cat or "jhumka" in fol):
                continue
                
            # Must strictly match the folder or category
            if cf != cat and cf != fol and cf not in fol:
                continue
                
        products_list.append({
            "id": jid,
            "name": info["name"],
            "category": info["category"],
            "folder": info["folder"],
            "image_path": info["image_path"],
            "price": 45000, 
            "type": info["type"]
        })
        
    return jsonify({"success": True, "products": products_list})


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN TRY-ON API WITH PERFORMANCE OPTIMIZATION
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
    global _frame_counter
    
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data"}), 400

        if data.get("action") == "start_tryon":
            _frame_counter = 0  # Reset frame counter
            return jsonify({"status": "success", "session_id": "123"})

        if data.get("action") == "process_frame" or \
           ("data" in data and "frame" in data.get("data", {})):

            # FIX: REMOVED FRAME SKIPPING - PROCESS EVERY FRAME
            _frame_counter += 1

            frame_data        = data["data"]["frame"]
            zoom              = float(data["data"].get("zoom_factor", 1.0))
            jewelry_ids       = data["data"].get("jewelry_ids", [])
            jewelry_id_single = data["data"].get("jewelry_id")
            camera_is_usb     = bool(data["data"].get("camera_is_usb", False))

            if not jewelry_ids and jewelry_id_single:
                jewelry_ids = [int(jewelry_id_single)]
            elif isinstance(jewelry_ids, list):
                jewelry_ids = [int(x) for x in jewelry_ids]
            else:
                jewelry_ids = [int(jewelry_ids)]

            frame = decode_base64_image(frame_data)
            if frame is None:
                return jsonify({"status": "error", "message": "Bad frame"})

            original_frame = frame.copy()
            t_start        = time.time()
            rgb            = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── Face detection ─────────────────────────────────────────────
            face_results = face_detector.process(rgb)

            # ── Hand detection ─────────────────────────────────────────────
            has_ring_item = any(
                jewelry_dataset.get(jid, {}).get("type") in ("ring", "bracelet")
                for jid in jewelry_ids
            )
            hand_results = None
            if has_ring_item:
                hand_results = hand_detector.process(rgb)

            person_detected    = False
            status             = "No Face"
            confidence         = 0.0
            first_jewelry_type = "necklace"
            first_category     = "necklace"
            hand_detected      = False

            face_region = frame[frame.shape[0]//4 : frame.shape[0]*3//4,
                                frame.shape[1]//4 : frame.shape[1]*3//4]
            brightness  = float(cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY).mean())
            lighting_ok = brightness > 60

            if face_results.multi_face_landmarks:
                person_detected = True
                face_lm         = face_results.multi_face_landmarks[0]
                lm_in_frame     = sum(1 for l in face_lm.landmark
                                      if 0.0 <= l.x <= 1.0 and 0.0 <= l.y <= 1.0)
                confidence      = lm_in_frame / len(face_lm.landmark)
            else:
                face_lm = None

            if hand_results is not None and hand_results.multi_hand_landmarks:
                hand_detected = True

            # CRITICAL FIX: Apply jewelry to EVERY frame
            for j_id in jewelry_ids:
                frame, status, success = apply_virtual_jewelry(
                    frame, face_lm, hand_results, j_id, zoom
                )
                if j_id in jewelry_dataset:
                    first_jewelry_type = jewelry_dataset[j_id]["type"]
                    first_category     = jewelry_dataset[j_id]["category"]
                    
                # DEBUG: Print if jewelry was applied
                print(f"[DEBUG] Jewelry {j_id} applied: {success}, status: {status}")

            if not person_detected and hand_detected:
                person_detected = True

            inference_ms = (time.time() - t_start) * 1000

            quality_score = 0.72
            if accuracy_scorer is not None and person_detected:
                quality_score = accuracy_scorer.predict(frame, first_category)

            acc_result = vto_tracker.record_frame(
                face_landmarks       = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None,
                face_detected        = person_detected,
                detection_confidence = confidence,
                original_frame       = original_frame,
                processed_frame      = frame,
                jewelry_type         = first_jewelry_type,
                category             = first_category,
                frame_shape          = frame.shape,
                zoom_factor          = zoom,
                inference_ms         = inference_ms,
            )

            combined_accuracy    = None
            live_metrics_payload = None
            if acc_result is not None and accuracy_scorer is not None:
                live_metrics      = vto_tracker.get_live_metrics()
                combined_accuracy = accuracy_scorer.compute_combined_score(live_metrics, quality_score)

            if vto_tracker.total_frames % 30 == 0:
                live_metrics_payload = vto_tracker.get_live_metrics()
                if combined_accuracy is not None:
                    live_metrics_payload["combined_accuracy"] = round(combined_accuracy, 3)
                    live_metrics_payload["quality_score"]     = round(quality_score, 3)

            if live_metrics_payload and vto_tracker.total_frames % 30 == 0:
                from flask import session
                session_id = session.get('vto_session_id')
                if session_id:
                    analytics.log_event('accuracy',
                        session_id=session_id,
                        pck_5pct=live_metrics_payload.get('pck_5pct', 0),
                        mean_iou=live_metrics_payload.get('mean_iou', 0),
                        mean_pos_error_px=live_metrics_payload.get('mean_pos_error_px', 0),
                        fps=live_metrics_payload.get('fps', 0),
                        detection_rate_pct=live_metrics_payload.get('detection_rate_pct', 0),
                        mean_jitter_px=live_metrics_payload.get('mean_jitter_px', 0),
                        overall_grade=live_metrics_payload.get('overall_grade', 'D'),
                        quality_score=live_metrics_payload.get('quality_score', 0)
                    )

            if not lighting_ok:
                status = "Low light — face a lamp or window"
            elif not person_detected:
                if has_ring_item:
                    status = "Show your hand to camera for ring try-on"
                else:
                    status = "No face detected"

            # ── Gaze ──────────────────────────────────────────────────────
            gaze_data = None
            if GAZE_ENABLED and gaze_detector_obj is not None and camera_is_usb:
                face_lm_gaze = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None
                h, w         = frame.shape[:2]
                gr           = gaze_detector_obj.update(face_lm_gaze, w, h)
                gaze_data    = {
                    "gaze_x":        gr["gaze_x"],
                    "gaze_y":        gr.get("gaze_y", 0.5),  # ✨ ADD gaze_y
                    "looking_at_y":  gr.get("gaze_y", 0.5),  # ✨ ADD looking_at_y for card targeting
                    "looking_right": gr["looking_right"],
                    "dwell_pct":     gr["dwell_pct"],
                    "dwell_event":   gr["dwell_event"],
                    "gaze_valid":    gr["gaze_valid"],
                }
            elif GAZE_ENABLED and not camera_is_usb:
                gaze_data = {
                    "gaze_x": 0.0,
                    "gaze_y": 0.5,
                    "looking_at_y": None,
                    "looking_right": False,
                    "dwell_pct": 0.0,
                    "dwell_event": False,
                    "gaze_valid": False,
                    "usb_required": True,
                }

            # CRITICAL: ALWAYS encode and return processed_frame ✨✨✨
            encoded_frame = encode_image_to_base64(frame)
            
            # DEBUG: Check if encoding succeeded
            if not encoded_frame:
                print("[DEBUG] ❌ Frame encoding FAILED!")
            else:
                print(f"[DEBUG] ✅ Frame encoded successfully, length: {len(encoded_frame)}")

            return jsonify({
                "processed_frame": encoded_frame,  # ✨ ALWAYS INCLUDED
                "position_status": status,
                "person_detected": person_detected,
                "hand_detected":   hand_detected,
                "confidence":      round(confidence, 3),
                "jewelry_count":   len(jewelry_ids),
                "ml_model_active": ML_MODELS_READY,
                "live_accuracy":   live_metrics_payload,
                "lighting_ok":     lighting_ok,
                "brightness":      round(brightness, 1),
                "gaze":            gaze_data,
                "camera_is_usb":   camera_is_usb,
                "frame_number":    _frame_counter,
                "fps":             round(1000 / max(inference_ms, 1), 1),
                "status":          "success",
            })

    except Exception as e:
        print(f"[API Error] {e}")
        import traceback; traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

    return jsonify({"status": "error", "message": "Unknown action"})


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — RECOMMENDATION API WITH CHAIN FILTERING
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/api/recommendations/<int:jewelry_id>")
def get_recommendations_api(jewelry_id):
    if jewelry_id not in jewelry_dataset:
        return jsonify({"success": False, "recommendations": [], "count": 0})
    info = jewelry_dataset[jewelry_id]
    
    # FIX: Gender and type-aware filtering
    jewelry_type = info["type"]
    category = info["category"]
    
    # Determine gender context
    is_mens = category in ['for-him', 'chain'] or 'men' in category.lower()
    
    # FIX: For chains, only recommend chains
    if jewelry_type == "chain":
        valid_types = ["chain"]
        exclude_categories = ["for-her", "wedding", "bridal", "jhumka", "earring"]
    else:
        valid_types = None
        exclude_categories = []

    if rec_engine is not None:
        try:
            query_path = os.path.join(app.static_folder, info["image_path"])
            recs = rec_engine.recommend(query_path=query_path, query_category=info["category"], top_k=12)
            
            # Filter recommendations
            filtered_recs = []
            for r in recs:
                rec_cat = r.get("category", "")
                rec_path = r.get("path", "")
                rec_jid = jewelry_by_path.get(rec_path)
                rec_type = jewelry_dataset.get(rec_jid, {}).get("type") if rec_jid else None
                
                # Skip excluded categories
                if any(exc in rec_cat for exc in exclude_categories):
                    continue
                
                # Type filter for chains
                if valid_types and rec_type not in valid_types:
                    continue
                
                # Gender filter for men's items
                if is_mens and any(fem in rec_cat for fem in ['for-her', 'bridal', 'wedding']):
                    continue
                
                filtered_recs.append(r)
                if len(filtered_recs) >= 8:
                    break
            
            return jsonify({
                "success": True, "engine": "clip_gnn_filtered",
                "recommendations": [{
                    "id":         jewelry_by_path.get(r["path"]),
                    "name":       r["name"],
                    "category":   r["category"],
                    "image_path": r["path"].replace(app.static_folder, "").lstrip("/\\"),
                    "clip_sim":   round(r["clip_sim"], 3),
                    "gnn_sim":    round(r["gnn_sim"], 3),
                    "score":      round(r["final_score"], 3),
                    "reason":     f"Style compatibility: {r['rule_compat']:.0%}",
                } for r in filtered_recs],
                "count": len(filtered_recs),
            })
        except Exception as e:
            print(f"[Rec] CLIP+GNN failed: {e}, falling back.")

    try:
        from recommendations import recommender
        from models import Product
        product = Product.query.filter_by(image_path=info["image_path"]).first()
        if product:
            recs = recommender.get_recommendations(product.id, max_results=8)
            return jsonify({
                "success": True, "engine": "rule_based",
                "recommendations": [{
                    "id":         r["product"]["id"],
                    "name":       r["product"]["name"],
                    "image_path": r["product"]["image_path"],
                    "category":   r["product"]["category"],
                    "score":      r["score"],
                    "similarity": r["similarity"],
                    "reason":     r["reason"],
                } for r in recs],
                "count": len(recs),
            })
    except Exception as e:
        print(f"[Rec] Rule-based fallback also failed: {e}")

    return jsonify({"success": False, "recommendations": [], "count": 0})


@app.route("/api/similar/<int:jewelry_id>")
def get_similar_api(jewelry_id):
    if jewelry_id not in jewelry_dataset:
        return jsonify({"success": False, "similar": []})
    info = jewelry_dataset[jewelry_id]
    try:
        from recommendations import recommender
        from models import Product
        product = Product.query.filter_by(image_path=info["image_path"]).first()
        if product:
            similar = recommender.get_similar_products(product.id, max_results=4)
            return jsonify({"success": True, "similar": [{
                "id": s["product"]["id"], "name": s["product"]["name"],
                "image_path": s["product"]["image_path"],
                "category": s["product"]["category"], "similarity": s["similarity"],
            } for s in similar]})
    except Exception as e:
        print(f"[Similar] Error: {e}")
    return jsonify({"success": False, "similar": []})


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9 — ACCURACY API
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/api/vto/accuracy/start", methods=["POST"])
def vto_accuracy_start():
    data = request.json or {}
    vto_tracker.reset()
    vto_tracker.category_tested = data.get("category", "unknown")
    return jsonify({"success": True, "message": "Accuracy tracking started", "session_start": vto_tracker.session_start})

@app.route("/api/vto/accuracy/live", methods=["GET"])
def vto_accuracy_live():
    if vto_tracker.total_frames == 0:
        return jsonify({"success": False, "message": "No frames yet. Start a try-on session first."})
    metrics = vto_tracker.get_live_metrics()
    if accuracy_scorer is not None:
        metrics["quality_model_active"] = accuracy_scorer.is_loaded
        metrics["mean_inference_ms"]    = round(accuracy_scorer.mean_inference_ms, 1)
    return jsonify({"success": True, "metrics": metrics})

@app.route("/api/vto/accuracy/report", methods=["GET"])
def vto_accuracy_report():
    if vto_tracker.total_frames == 0:
        return jsonify({"success": False, "message": "No frames recorded yet."})
    return jsonify({"success": True, "report": vto_tracker.get_accuracy_report()})

@app.route("/api/vto/accuracy/save", methods=["POST"])
def vto_accuracy_save():
    if vto_tracker.total_frames == 0:
        return jsonify({"success": False, "message": "No data to save"})
    report    = vto_tracker.get_accuracy_report()
    rdir      = os.path.join(app.static_folder, "reports")
    os.makedirs(rdir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"vto_accuracy_{timestamp}.json"
    with open(os.path.join(rdir, filename), "w") as f:
        json.dump(report, f, indent=2)
    return jsonify({"success": True, "filename": filename, "report": report,
                    "message": f"Saved to static/reports/{filename}"})

@app.route("/api/vto/accuracy/reset", methods=["POST"])
def vto_accuracy_reset():
    vto_tracker.reset()
    return jsonify({"success": True, "message": "Accuracy data cleared."})


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10 — STATUS + GAZE + CAMERA API ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/api/model/status")
def model_status():
    return jsonify({
        "ml_model_loaded":        ML_MODELS_READY,
        "overlay_engine":         overlay_engine is not None,
        "recommendation_engine":  rec_engine is not None,
        "accuracy_quality_model": accuracy_scorer.is_loaded if accuracy_scorer else False,
        "gaze_detector":          gaze_detector_obj is not None,
        "gaze_enabled":           GAZE_ENABLED,
        "device":                 str(DEVICE) if DEVICE else "cpu",
        "jewelry_items_loaded":   len(jewelry_dataset),
        "message": "Custom model active" if ML_MODELS_READY else "Using MediaPipe fallback",
    })


@app.route("/api/gaze/status", methods=["GET"])
def gaze_status():
    if gaze_detector_obj is None:
        return jsonify({"active": False})
    return jsonify({"active": True, "stats": gaze_detector_obj.get_stats()})


@app.route("/api/gaze/toggle", methods=["POST"])
def gaze_toggle():
    global GAZE_ENABLED
    data         = request.json or {}
    requested    = data.get("enabled", not GAZE_ENABLED)
    camera_is_usb = data.get("camera_is_usb", False)

    if requested and not camera_is_usb:
        return jsonify({
            "gaze_enabled": False,
            "blocked": True,
            "message": "Gaze control requires USB camera. Please select a USB camera (index 1+) first."
        })

    GAZE_ENABLED = requested
    if gaze_detector_obj:
        gaze_detector_obj.is_active = GAZE_ENABLED
    return jsonify({
        "gaze_enabled": GAZE_ENABLED,
        "blocked": False,
        "message": "Gaze enabled" if GAZE_ENABLED else "Gaze disabled"
    })


@app.route("/api/gaze/debug", methods=["GET"])
def gaze_debug():
    if gaze_detector_obj is None:
        return jsonify({
            "error": "Gaze detector not loaded",
            "fix":   "Check terminal for [GAZE] line when app starts"
        })

    stats = gaze_detector_obj.get_stats()
    gx    = stats["current_offset"] if "current_offset" in stats else stats.get("current_gaze_x", 0)
    thr   = stats["threshold"]
    valid = stats.get("iris_valid", False)

    diagnosis = []
    if not valid:
        diagnosis.append("PROBLEM: Iris landmarks not detected (invalid=True)")
        diagnosis.append("FIX: Make sure refine_landmarks=True in face_detector")
        diagnosis.append("FIX: Move face closer to camera (< 80cm)")
    else:
        diagnosis.append(f"gaze_x={gx:.3f} — iris detected and moving")

    return jsonify({
        "iris_valid":                valid,
        "current_gaze_x":            round(gx, 4),
        "currently_detecting_right": gx < thr,
        "dwell_pct":                 stats["current_dwell_pct"],
        "current_threshold":         thr,
        "method":                    stats.get("method", "head_pose_primary"),
        "diagnosis":                 diagnosis,
        "note":                      "Gaze only works with USB camera (index 1+)",
        "stats":                     stats,
    })


@app.route("/api/camera/list", methods=["GET"])
def camera_list():
    cameras = []
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cameras.append({
                "index":       i,
                "resolution":  f"{w}x{h}",
                "label":       "Built-in webcam" if i == 0 else f"USB Camera {i}",
                "is_usb":      i > 0,
                "current":     i == CAMERA_INDEX,
                "gaze_compatible": i > 0,
            })
            cap.release()
    return jsonify({
        "cameras":        cameras,
        "current_index":  CAMERA_INDEX,
        "gaze_note":      "Gaze control only works with USB cameras (index 1+)"
    })


@app.route("/api/camera/set", methods=["POST"])
def camera_set():
    global CAMERA_INDEX, GAZE_ENABLED
    data    = request.json or {}
    new_idx = int(data.get("index", 0))
    CAMERA_INDEX = new_idx

    if new_idx == 0 and GAZE_ENABLED:
        GAZE_ENABLED = False
        if gaze_detector_obj:
            gaze_detector_obj.is_active = False
        gaze_auto_disabled = True
    else:
        gaze_auto_disabled = False

    return jsonify({
        "success":            True,
        "new_index":          CAMERA_INDEX,
        "is_usb":             new_idx > 0,
        "gaze_auto_disabled": gaze_auto_disabled,
        "message":            f"Camera {CAMERA_INDEX} selected. {'Gaze auto-disabled (built-in camera).' if gaze_auto_disabled else ''}"
    })

@app.route("/api/analytics/session/start", methods=["POST"])
def analytics_session_start():
    data = request.json or {}
    session_id = generate_session_id()
    
    from flask import session
    from flask_login import current_user
    
    analytics.log_event('session_start',
        session_id=session_id,
        user_id=current_user.id if current_user.is_authenticated else None,
        camera_type=data.get('camera_type', 'builtin'),
        gaze_enabled=data.get('gaze_enabled', False)
    )
    
    session['vto_session_id'] = session_id
    
    return jsonify({"success": True, "session_id": session_id})

@app.route("/api/analytics/session/end", methods=["POST"])
def analytics_session_end():
    from flask import session
    data = request.json or {}
    session_id = data.get('session_id') or session.get('vto_session_id')
    
    if not session_id:
        return jsonify({"success": False, "message": "No session found"})
    
    analytics.log_event('session_end',
        session_id=session_id,
        duration_sec=data.get('duration_sec', 0),
        total_frames=data.get('total_frames', 0)
    )
    
    session.pop('vto_session_id', None)
    return jsonify({"success": True})


@app.route("/api/analytics/event", methods=["POST"])
def analytics_log_event():
    data = request.json or {}
    event_type = data.pop('event_type', 'unknown')
    
    analytics.log_event(event_type, **data)
    return jsonify({"success": True})


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11 — STARTUP
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    load_jewelry_dataset()
    load_ml_models()
    
    analytics.start()
    
    with app.app_context():
        try:
            from recommendations import recommender
            from models import Product
            all_products = Product.query.filter_by(is_active=True).all()
            recommender.build_index(all_products, app.static_folder)
        except Exception as e:
            print(f"[Rec] Rule-based index: {e}")
    
    try:
        app.run(host="0.0.0.0", debug=True, port=5000, use_reloader=False)
    finally:
        analytics.stop() 