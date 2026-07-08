# Project Documentation — Parshva Jewellers AI Virtual Try-On & Recommendation System

## 1. Overview

This project is a jewelry e-commerce web app built around two AI features:
- A **virtual try-on** system that overlays jewelry onto a live webcam feed
- A **recommendation engine** that suggests complementary jewelry pieces using a
  combination of visual embeddings and learned pairing patterns

The project category is **Recommendation**, with virtual try-on (computer vision) as a
supporting feature.

## 2. Architecture

Three containers, connected over a custom Docker network:

```
Browser ──▶ frontend (React, port 5173) ──▶ backend (Flask, port 5000) ──▶ db (MySQL)
```

- **frontend**: React + Vite. Handles all page rendering and routing (React Router)
  client-side. Talks to the backend exclusively through `fetch()` calls to `/api/*`
  endpoints — no server-rendered pages.
- **backend**: Flask. Owns all business logic — auth, catalog, cart/wishlist, the
  try-on/overlay pipeline, and the recommendation engine. Only the backend talks to MySQL.
- **db**: MySQL 8. Stores users, OTP records, products, cart items, and wishlist items.

## 3. Core Features

### 3.1 Virtual Try-On
- MediaPipe detects face and hand landmarks in each webcam frame (sent from the
  browser as a base64 JPEG).
- An overlay engine (`ml_models/overlay_engine.py`) positions the jewelry image based on
  landmark geometry — different logic for necklaces/chains/mangalsutras (face-anchored),
  rings (hand-anchored), and earrings.
- Supports overlaying **multiple jewelry items simultaneously** ("Complete Your Look"),
  via a `jewelry_ids` array in the try-on request.

### 3.2 Recommendation Engine
Two systems work together:
- **CLIP + GNN engine** (`ml_models/advanced_recommendation.py`): a CLIP (ViT-B/32)
  visual embedding captures what a piece looks like; a GNN layer is trained on which
  pieces are actually paired/bought together, so recommendations reflect real pairing
  patterns, not just visual similarity.
- **Rule-based fallback** (`recommendations.py`): color histogram + texture + material
  matching, used as a simpler complementary-category recommender.

### 3.3 Accuracy Scoring
- Rule-based geometric metrics (PCK, IoU, positional error, frame-to-frame jitter) are
  computed every frame by `vto_accuracy.py`.
- A learned EfficientNet-B0 model (`ml_models/accuracy_model.py`) separately scores
  overall visual/perceptual quality of the render (things geometry alone can't catch,
  like lighting or perspective mismatch).
- The two scores are combined into a single accuracy grade (A–D).

### 3.4 E-commerce shell
Catalog browsing (by material, collection, wedding, gifting), search, cart, wishlist,
phone + OTP authentication, and an admin panel for managing products.

## 4. Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask, Flask-SQLAlchemy, Flask-Login |
| ML / CV | PyTorch, timm (EfficientNet-B0), CLIP (ViT-B/32), custom GNN, MediaPipe, OpenCV |
| Database | MySQL 8 |
| Frontend | React 19, Vite, React Router, Tailwind CSS |
| Infrastructure | Docker, Docker Compose |

## 5. Key API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/products` | GET | List/filter catalog products (`?category=`) |
| `/api/jewelry-tryon` | POST | Process a webcam frame, return the try-on render |
| `/api/recommendations/<id>` | GET | Get AI recommendations for a product |
| `/api/auth/signup` / `/login` | POST | Start phone + OTP authentication |
| `/api/auth/verify-otp` | POST | Verify OTP, create session |
| `/api/auth/status` | GET | Check current login state |
| `/api/cart/add`, `/api/wishlist/toggle` | POST | Cart/wishlist mutations |
| `/api/admin/product/*` | POST | Admin product management |

## 6. Data

Product images live under `static/`, organized by category folder (`necklace/`, `ring/`,
`chain/`, `jhumka/`, `mangalsutra/`, plus collection/wedding/gifting groupings). Product
rows are created automatically at startup by scanning these folders — no manual database
seeding is required.

## 7. Data Preparation (for model training)

Training the accuracy and recommendation models used more examples than the raw product
photos provide:
- Classical augmentation (rotation, lighting, cropping) expanded the training set.
- Synthetic try-on frames were generated **offline**, using diffusion-based image
  generation, to give the accuracy model both good and bad placement examples to learn
  the difference between.

None of this generation happens at runtime — it was a one-time step to build the training
set before the models were trained; the deployed app only runs inference.

## 8. Known Limitations

- Trained model checkpoints are included in the repository as binary files; a very
  constrained environment (e.g. extremely limited disk/bandwidth) may take longer to
  clone as a result.
- The rule-based fallback recommender and the CLIP+GNN engine use independent internal
  ID spaces, bridged via image path lookups rather than shared IDs by design.
