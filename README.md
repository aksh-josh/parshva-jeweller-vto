# Parshva Jewellers — AI Virtual Try-On & Recommendation System

Parshva Jewellers is a jewelry web app where you can browse a catalog, try pieces on
virtually through your webcam, and get AI-recommended pieces that go well with whatever
you're looking at — a necklace suggesting matching earrings, for example.

## Why I made this

Online jewelry shopping has two problems that never felt solved to me: you can't tell how
something will actually look on you, and there's no good way to know what else would pair
with it. Most e-commerce recommendation is just "customers who viewed this also viewed
that" — it doesn't actually understand what makes two pieces go together.

So I built two things on top of a normal jewelry catalog:
- a virtual try-on that overlays jewelry onto your live webcam feed using face/hand
  landmark detection, so you can actually see it on yourself before buying
- a recommendation engine that combines a CLIP visual embedding (what does it look like)
  with a GNN trained on how pieces are actually paired (what goes with what), instead of
  just nearest-neighbour similarity

I also built a learned accuracy model that scores how visually convincing each try-on
render is, on top of simpler geometric checks (landmark position error, overlap, jitter
between frames) — so the system isn't just guessing where to place things, it has a way of
telling how well it's doing.

## How it works

Three parts, each in its own Docker container:
- **frontend**: a React (Vite) app — catalog browsing, the try-on camera page, cart,
  wishlist, login, admin panel
- **backend**: a Python (Flask) API — serves the catalog, runs the try-on/overlay engine,
  the recommendation engine, and auth
- **database**: MySQL, storing users, products, cart, and wishlist

The frontend never talks to the database directly. It calls the backend's API (things
like `/api/products`, `/api/jewelry-tryon`, `/api/recommendations/<id>`, `/api/cart/add`),
and the backend is the only thing that touches MySQL. The three containers are connected
over a custom Docker network.

For the try-on itself, MediaPipe detects face and hand landmarks in each webcam frame, and
an overlay engine positions the jewelry image accordingly (different logic for necklaces,
rings, and earrings). For recommendations, a CLIP (ViT-B/32) embedding captures visual
style, and a GNN layer is trained on which pieces actually get worn/bought together, so a
recommendation isn't just "looks similar" — it's "looks similar to what people actually
pair this with." An EfficientNet-B0 model separately scores the quality of each try-on
render, combined with rule-based geometric metrics into a single accuracy score.

## How to run it

You need Docker Desktop installed and running. From the project folder run:

```
docker compose up --build
```

Then open:
- the app: http://localhost:5173
- the backend API directly (for debugging): http://localhost:5000

The database and product catalog set themselves up the first time the backend starts —
tables are created automatically, and product entries are imported from the images in
`static/`, so there's nothing else to configure.

## How to use it

1. Browse the catalog by material (gold, silver, diamond, daily wear) or by collection.
2. Click a piece you like, then "Virtual Try-On" to open the camera view.
3. See the piece rendered live on your camera feed, adjust zoom if needed.
4. Check the recommendations panel for pieces that pair well with what you're wearing.
5. Add pieces to your cart or wishlist, or sign in (phone number + OTP) to save them.

## The data

The catalog is a set of jewelry product images organized by category (necklace, ring,
chain, jhumka, mangalsutra, bangles, and various collection/wedding/gifting groupings)
under `static/`. Each image becomes a product row in the database automatically.

## How I prepared the data

For the recommendation engine and the accuracy model, I needed more training examples
than the raw product photos alone, so:
- classical augmentation (rotation, lighting, cropping) expanded the image set for training
- synthetic try-on frames were generated offline (using diffusion-based image generation)
  to give the accuracy model examples of both good and bad jewelry placement to learn from

None of that generation happens at runtime — it was a one-time offline step to build the
training set; the deployed app only does inference with the already-trained models.

## Tools I used

- Python and Flask for the backend and API
- PyTorch, timm (EfficientNet-B0), CLIP (ViT-B/32), and a custom GNN for the ML models
- MediaPipe and OpenCV for face/hand landmark detection and image processing
- MySQL for the database
- React (with Vite) and Tailwind CSS for the frontend
- Docker and Docker Compose to run everything together

## Project structure

```
AI_VTO_Project - Copyyy/
  docker-compose.yml       starts the three containers on a custom network
  AI_VTO_Project/          backend
    final_app.py           main Flask app — routes, try-on, recommendations
    models.py               database models (users, products, cart, wishlist)
    auth.py                 phone + OTP authentication
    shop.py                 catalog, cart, wishlist, search endpoints
    admin.py                admin product management endpoints
    recommendations.py      rule-based recommendation fallback
    vto_accuracy.py          geometric accuracy tracking for try-on frames
    site_structure.py       category/collection/material lookup tables
    ml_models/
      advanced_recommendation.py   CLIP + GNN recommendation engine
      overlay_engine.py            jewelry placement/overlay logic
      accuracy_model.py            EfficientNet-B0 quality scoring model
      accuracy_inference.py        inference wrapper for the quality model
    static/                 product images, organized by category
    checkpoints/            trained model weights
    Dockerfile
    requirements.txt
  frontend/
    src/
      App.jsx                routes
      components/Layout.jsx   shared nav/footer, auth state
      pages/                  Home, Catalog, VirtualTryOn, Auth, Cart, Wishlist, Profile, Admin
    Dockerfile
    vite.config.js
    package.json
```
