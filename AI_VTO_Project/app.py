import os
from flask import Flask, render_template, url_for, request, jsonify
from flask_cors import CORS
from flask_login import LoginManager
from recommendations import recommender
from vto_accuracy import VTOAccuracyTracker
import cv2
import numpy as np
import base64
import mediapipe as mp
import time
import random
import json
from datetime import datetime

import config
from models import db, User, init_db
from auth import auth_bp
from shop import shop_bp
from admin import admin_bp

app = Flask(__name__)
CORS(app)

# -------------------------------------------------------------------
# APP CONFIG
# -------------------------------------------------------------------
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = config.SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = config.SQLALCHEMY_TRACK_MODIFICATIONS
app.config['MYSQL_HOST'] = config.MYSQL_HOST
app.config['MYSQL_USER'] = config.MYSQL_USER
app.config['MYSQL_PASSWORD'] = config.MYSQL_PASSWORD
app.config['MYSQL_DATABASE'] = config.MYSQL_DATABASE

# -------------------------------------------------------------------
# DATABASE + AUTH INIT
# -------------------------------------------------------------------
init_db(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.auth_page'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(shop_bp)
app.register_blueprint(admin_bp)

# -------------------------------------------------------------------
# 1. AI SETUP & GLOBAL VARIABLES
# -------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_detector = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

jewelry_dataset = {}
vto_tracker = VTOAccuracyTracker()

# -------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -------------------------------------------------------------------

def remove_white_background(image, threshold=240):
    """Makes white/near-white pixels transparent."""
    if image is None:
        return None
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    white_pixels = np.where(
        (image[:, :, 0] > threshold) &
        (image[:, :, 1] > threshold) &
        (image[:, :, 2] > threshold)
    )
    image[white_pixels[0], white_pixels[1], 3] = 0
    return image


def get_images(folder_name):
    folder_path = os.path.join(app.static_folder, folder_name)
    images = []
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")) and filename.lower() != 'cover.jpeg':
                images.append(f"{folder_name}/{filename}")
    return images


def load_jewelry_dataset():
    global jewelry_dataset
    static_path = app.static_folder
    jewelry_id = 1

    folder_mapping = {
        'necklace': 'necklace',
        'mangalsutra': 'mangalsutra',
        'rajwadi': 'rajwadi',
        'jhumka': 'jhumka',
        'earrings': 'jhumka',
        'gifting': 'gifting',
        'wedding': 'wedding',
        'chain': 'chain',
        'ring': 'ring',
        'bangles': 'bangles'
    }

    for category_key, folder_name in folder_mapping.items():
        folder_path = os.path.join(static_path, folder_name)

        if 'ring' in folder_name:
            vto_type = 'ring'
        elif 'bangles' in folder_name:
            vto_type = 'bracelet'
        elif 'jhumka' in folder_name or 'earrings' in category_key:
            vto_type = 'earring'
        elif 'chain' in folder_name:
            vto_type = 'chain'
        else:
            vto_type = 'necklace'

        if not os.path.exists(folder_path):
            continue

        for image_file in os.listdir(folder_path):
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                continue

            full_path = os.path.join(folder_path, image_file)
            relative_path = f"{folder_name}/{image_file}"
            img_data = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

            if img_data is not None:
                if full_path.lower().endswith(('.jpg', '.jpeg')) or img_data.shape[2] == 3:
                    img_data = remove_white_background(img_data)

                jewelry_dataset[jewelry_id] = {
                    "id": jewelry_id,
                    "type": vto_type,
                    "name": os.path.splitext(image_file)[0].replace('_', ' ').title(),
                    "category": category_key,
                    "folder": folder_name,
                    "image_path": relative_path,
                    "image_data": img_data,
                    "size_factor": 1.0
                }
                jewelry_id += 1

    print(f"Loaded {len(jewelry_dataset)} jewelry items.")


# -------------------------------------------------------------------
# 3. AI LOGIC
# -------------------------------------------------------------------

def decode_base64_image(base64_string):
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(base64_string), np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def encode_image_to_base64(image):
    try:
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('ascii')}"
    except Exception:
        return None


def overlay_image(background, overlay, x, y):
    """Alpha-blend overlay onto background using float32 (no overflow)."""
    bg_h, bg_w = background.shape[:2]
    ol_h, ol_w = overlay.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + ol_w), min(bg_h, y + ol_h)
    ol_x1, ol_y1 = max(0, -x), max(0, -y)
    ol_x2, ol_y2 = ol_x1 + (x2 - x1), ol_y1 + (y2 - y1)

    if x2 <= x1 or y2 <= y1:
        return background

    overlay_crop = overlay[ol_y1:ol_y2, ol_x1:ol_x2]
    bg_region = background[y1:y2, x1:x2]

    if overlay_crop.shape[2] == 4:
        alpha = overlay_crop[:, :, 3].astype(np.float32) / 255.0
        alpha_3d = alpha[:, :, np.newaxis]
        fg = overlay_crop[:, :, :3].astype(np.float32)
        bg_f = bg_region.astype(np.float32)
        blended = (alpha_3d * fg + (1.0 - alpha_3d) * bg_f)
        np.clip(blended, 0, 255, out=blended)
        background[y1:y2, x1:x2] = blended.astype(np.uint8)
    else:
        background[y1:y2, x1:x2] = overlay_crop[:, :, :3]

    return background


def get_lm_px(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)


def apply_virtual_jewelry(image, face_landmarks, jewelry_id, zoom_factor=1.0):
    if jewelry_id not in jewelry_dataset:
        return image, "Invalid ID", False

    info = jewelry_dataset[jewelry_id]
    jewelry_type = info["type"]
    jewelry_img = info["image_data"]

    if jewelry_img is None:
        return image, "Image error", False

    h_img, w_img = image.shape[:2]
    lm = face_landmarks.landmark

    # Face width
    left_x, _ = get_lm_px(lm[234], w_img, h_img)
    right_x, _ = get_lm_px(lm[454], w_img, h_img)
    face_width = abs(right_x - left_x)

    if face_width < 20:
        return image, "Face too small", False

    # --- NECKLACE / CHAIN / MANGALSUTRA ---
    if jewelry_type in ("necklace", "chain", "mangalsutra"):
        chin_x, chin_y = get_lm_px(lm[152], w_img, h_img)

        target_width = int(face_width * 1.7 * zoom_factor)
        target_width = max(60, target_width)

        ol_h, ol_w = jewelry_img.shape[:2]
        aspect = ol_h / ol_w
        target_height = int(target_width * aspect)

        resized = cv2.resize(jewelry_img, (target_width, target_height), interpolation=cv2.INTER_AREA)

        pos_x = chin_x - target_width // 2
        pos_y = chin_y + int(face_width * 0.2)

        image = overlay_image(image, resized, pos_x, pos_y)
        return image, "Necklace Placed", True

    # --- EARRING / JHUMKA ---
    elif jewelry_type == "earring":
        left_ear_x, left_ear_y = get_lm_px(lm[234], w_img, h_img)
        right_ear_x, right_ear_y = get_lm_px(lm[454], w_img, h_img)
        _, left_jaw_y = get_lm_px(lm[132], w_img, h_img)
        _, right_jaw_y = get_lm_px(lm[361], w_img, h_img)

        left_lobe_y = (left_ear_y + left_jaw_y) // 2
        right_lobe_y = (right_ear_y + right_jaw_y) // 2

        ol_h, ol_w = jewelry_img.shape[:2]
        img_aspect = ol_w / ol_h

        if img_aspect >= 0.85:
            mid = ol_w // 2
            left_earring_img = jewelry_img[:, :mid]
            right_earring_img = jewelry_img[:, mid:]
        else:
            left_earring_img = jewelry_img
            right_earring_img = cv2.flip(jewelry_img, 1)

        le_h, le_w = left_earring_img.shape[:2]
        re_h, re_w = right_earring_img.shape[:2]

        target_h = int(face_width * 0.45 * zoom_factor)
        target_h = max(30, target_h)

        left_target_w = max(15, int(target_h * (le_w / le_h)))
        right_target_w = max(15, int(target_h * (re_w / re_h)))

        resized_left = cv2.resize(left_earring_img, (left_target_w, target_h), interpolation=cv2.INTER_AREA)
        resized_right = cv2.resize(right_earring_img, (right_target_w, target_h), interpolation=cv2.INTER_AREA)

        lx = left_ear_x - left_target_w // 2
        ly = left_lobe_y
        image = overlay_image(image, resized_left, lx, ly)

        rx = right_ear_x - right_target_w // 2
        ry = right_lobe_y
        image = overlay_image(image, resized_right, rx, ry)

        return image, "Earrings Placed", True

    return image, "Unknown Type", False


# -------------------------------------------------------------------
# 4. WEB ROUTES
# -------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/collections/<collection_name>")
def collections(collection_name):
    images = get_images(collection_name)
    if collection_name == 'wedding':
        return render_template("necklace.html", images=images, title="Wedding Collection")
    elif collection_name == 'gifting':
        return render_template("gifting.html", images=images, title="Gifting Collection")
    return render_template("necklace.html", images=images, title=f"{collection_name.title()} Collection")

@app.route("/shop/<category>")
def shop_category(category):
    category_map = {
        'earrings': 'jhumka',
        'necklaces': 'necklace',
        'rajwadi': 'rajwadi',
        'mangalsutra': 'mangalsutra',
        'rings': 'ring',
        'bangles': 'bangles',
        'chain': 'chain'
    }
    folder_name = category_map.get(category)
    if not folder_name:
        return "Category not found", 404

    images = get_images(folder_name)
    title = category.replace('_', ' ').title()

    template_name = f"{folder_name}.html"
    if not os.path.exists(os.path.join(app.template_folder, template_name)):
        return render_template("necklace.html", images=images, title=title)
    return render_template(template_name, images=images, title=title)

@app.route("/tryon/<category>/<filename>")
def tryon(category, filename):
    found_id = None
    for jid, info in jewelry_dataset.items():
        if info["category"] == category and info["image_path"].endswith(filename):
            found_id = jid
            break
    if not found_id:
        return "Product not found. Please ensure the image exists in static folder.", 404

    return render_template("tryon_live.html",
                           category=category,
                           filename=filename,
                           jewelry_id=found_id,
                           jewelry_name=jewelry_dataset[found_id]['name'])

# Legacy routes
@app.route("/mangalsutra")
def mangalsutra():
    return render_template("mangalsutra.html", images=get_images("mangalsutra"), title="Mangalsutra")

@app.route("/jhumka")
def jhumka():
    return render_template("jhumka.html", images=get_images("jhumka"), title="Jhumka")

@app.route("/necklace")
def necklace():
    return render_template("necklace.html", images=get_images("necklace"), title="Necklace")

@app.route("/chain")
def chain():
    return render_template("chain.html", images=get_images("chain"), title="Chain")

@app.route("/ring")
def ring():
    return render_template("ring.html", images=get_images("ring"), title="Ring")


# -------------------------------------------------------------------
# 5. JEWELRY TRY-ON API
# -------------------------------------------------------------------

@app.route('/api/jewelry/categories', methods=['GET'])
def get_categories_api():
    serializable = []
    for k, v in jewelry_dataset.items():
        item = v.copy()
        del item['image_data']
        serializable.append(item)
    return jsonify({'status': 'success', 'categories': {'all': serializable}})


@app.route('/api/jewelry-tryon', methods=['POST'])
def api_tryon():
    try:
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': 'No data'}), 400

        # Handle 'start_tryon'
        if data.get('action') == 'start_tryon':
            return jsonify({'status': 'success', 'session_id': '123'})

        # Handle 'process_frame' AND legacy format (unified handler)
        if data.get('action') == 'process_frame' or ('data' in data and 'frame' in data.get('data', {})):
            frame_data = data['data']['frame']
            zoom = float(data['data'].get('zoom_factor', 1.0))

            # Get ALL jewelry IDs to render simultaneously
            jewelry_ids = data['data'].get('jewelry_ids', [])
            jewelry_id_single = data['data'].get('jewelry_id')

            if not jewelry_ids and jewelry_id_single:
                jewelry_ids = [int(jewelry_id_single)]
            elif isinstance(jewelry_ids, list):
                jewelry_ids = [int(x) for x in jewelry_ids]
            else:
                jewelry_ids = [int(jewelry_ids)]

            frame = decode_base64_image(frame_data)
            if frame is None:
                return jsonify({'status': 'error', 'message': 'Bad frame'})

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb)

            person_detected = False
            status = "No Face"
            confidence = 0.0

            if results.multi_face_landmarks:
                person_detected = True
                face_lm = results.multi_face_landmarks[0]

                # Compute detection confidence
                # Face mesh landmarks don't have meaningful visibility scores,
                # so we compute confidence from landmark spatial consistency:
                # - How many landmarks are within the frame (0-1 range)
                # - How stable the key anchor points are
                lm_in_frame = sum(1 for l in face_lm.landmark if 0.0 <= l.x <= 1.0 and 0.0 <= l.y <= 1.0)
                total_lm = len(face_lm.landmark)
                confidence = lm_in_frame / total_lm if total_lm > 0 else 0.85

                # Record frame for accuracy tracking (server-side with REAL landmarks)
                vto_tracker.record_frame(
                    landmarks=face_lm,
                    face_detected=True,
                    detection_confidence=confidence
                )

                # Apply ALL jewelry items onto the same frame
                for j_id in jewelry_ids:
                    frame, status, success = apply_virtual_jewelry(
                        frame, face_lm, j_id, zoom
                    )
            else:
                # No face — still record for detection rate
                vto_tracker.record_frame(
                    landmarks=None,
                    face_detected=False,
                    detection_confidence=0.0
                )

            return jsonify({
                "processed_frame": encode_image_to_base64(frame),
                "position_status": status,
                "person_detected": person_detected,
                "confidence": confidence,
                "jewelry_count": len(jewelry_ids),
                "status": "success"
            })

    except Exception as e:
        print(f"API Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})

    return jsonify({'status': 'error', 'message': 'Unknown action'})


# -------------------------------------------------------------------
# 6. AI RECOMMENDATION API ROUTES
# -------------------------------------------------------------------

@app.route('/api/recommendations/<int:jewelry_id>')
def get_recommendations_api(jewelry_id):
    """Get AI-powered complementary recommendations for a jewelry item."""
    if jewelry_id in jewelry_dataset:
        info = jewelry_dataset[jewelry_id]
        image_path = info['image_path']
        from models import Product
        product = Product.query.filter_by(image_path=image_path).first()
        if product:
            recs = recommender.get_recommendations(product.id, max_results=8)
            return jsonify({
                'success': True,
                'recommendations': [
                    {
                        'id': r['product']['id'],
                        'name': r['product']['name'],
                        'image_path': r['product']['image_path'],
                        'category': r['product']['category'],
                        'material': r['product']['material'],
                        'price': r['product']['price'],
                        'display_price': r['product']['display_price'],
                        'score': r['score'],
                        'similarity': r['similarity'],
                        'reason': r['reason'],
                    }
                    for r in recs
                ],
                'count': len(recs),
            })
    return jsonify({'success': False, 'recommendations': [], 'count': 0})


@app.route('/api/similar/<int:jewelry_id>')
def get_similar_api(jewelry_id):
    """Get visually similar products in same category."""
    if jewelry_id in jewelry_dataset:
        info = jewelry_dataset[jewelry_id]
        from models import Product
        product = Product.query.filter_by(image_path=info['image_path']).first()
        if product:
            similar = recommender.get_similar_products(product.id, max_results=4)
            return jsonify({
                'success': True,
                'similar': [
                    {
                        'id': s['product']['id'],
                        'name': s['product']['name'],
                        'image_path': s['product']['image_path'],
                        'category': s['product']['category'],
                        'material': s['product']['material'],
                        'similarity': s['similarity'],
                    }
                    for s in similar
                ],
            })
    return jsonify({'success': False, 'similar': []})


# -------------------------------------------------------------------
# 7. VTO ACCURACY API ROUTES (server-side tracking)
# -------------------------------------------------------------------

@app.route('/api/vto/accuracy/start', methods=['POST'])
def vto_accuracy_start():
    """Start a new accuracy tracking session."""
    data = request.json or {}
    vto_tracker.reset()
    vto_tracker.category_tested = data.get('category', 'unknown')
    return jsonify({'success': True, 'message': 'Accuracy tracking started'})

@app.route('/api/vto/accuracy/report', methods=['GET'])
def vto_accuracy_report():
    """Get the full accuracy report for the current session."""
    if vto_tracker.total_frames == 0:
        return jsonify({'success': False, 'message': 'No frames recorded yet.'})
    report = vto_tracker.get_accuracy_report()
    return jsonify({'success': True, 'report': report})

@app.route('/api/vto/accuracy/save', methods=['POST'])
def vto_accuracy_save():
    """Save accuracy report to JSON file."""
    if vto_tracker.total_frames == 0:
        return jsonify({'success': False, 'message': 'No data to save'})
    report = vto_tracker.get_accuracy_report()
    reports_dir = os.path.join(app.static_folder, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'vto_accuracy_{timestamp}.json'
    filepath = os.path.join(reports_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)
    return jsonify({'success': True, 'message': f'Report saved to reports/{filename}', 'report': report})


# -------------------------------------------------------------------
# 8. RUN
# -------------------------------------------------------------------
if __name__ == "__main__":
    load_jewelry_dataset()

    # Build AI recommendation index
    with app.app_context():
        from models import Product
        all_products = Product.query.filter_by(is_active=True).all()
        recommender.build_index(all_products, app.static_folder)

    app.run(debug=True, port=5000)