# Installation Guide — Parshva Jewellers

## Prerequisites

- **Docker Desktop** installed and running (Windows/Mac: download from
  docker.com; make sure the app is actually open and shows "Engine running"
  before proceeding — a very common first error is Docker Desktop simply not
  being started yet).
- **Git** installed (`git --version` to check).
- At least ~10GB free disk space (PyTorch, CLIP, and Depth Anything model
  weights are downloaded during the first build).

## Step 1 — Clone the repository

```
git clone https://github.com/aksh-josh/parshva-jeweller-vto
cd parshva-jeweller-vto
```

## Step 2 — Project structure check

Confirm you have this layout (see `PROJECT_DOCUMENTATION.md` for the full tree):
```
<repo-root>/
├── docker-compose.yml
├── AI_VTO_Project/     (backend)
└── frontend/           (React app)
```
Run these commands **from `<repo-root>`** — the folder containing `docker-compose.yml`.

## Step 3 — Build and run

```
docker compose up --build
```

First run will take several minutes: MySQL image, Node packages, Python/PyTorch
packages, and ML model weights (CLIP ~338MB, Depth Anything ~100MB) all download around 3 -4 minutes.
Subsequent runs are much faster.

## Step 4 — Access the app

- Frontend: **http://localhost:5173**
- Backend API (for debugging): **http://localhost:5000**

The database and product catalog set themselves up automatically the first time the
backend starts — tables are created, and products are imported from the images in
`static/`. No manual database setup is needed.

## Stopping the app

```
docker compose down
```

To also clear the database contents (safe — the catalog re-imports automatically on
next start):
```
docker compose down -v
```

## Troubleshooting — issues we hit during development, and their fixes

**"failed to connect to the docker API ... npipe" error**
Docker Desktop isn't running. Open the Docker Desktop application and wait until it
shows a running/green status before retrying.

**Container name conflict (e.g. "parshva_db" already in use)**
A leftover container from a previous run wasn't cleaned up.
```
docker rm -f parshva_db
docker compose up
```

**`RuntimeError: 'cryptography' package is required for ... caching_sha2_password`**
MySQL 8's default authentication method needs Python's `cryptography` package. This is
already included in `requirements.txt` — if you see this, confirm the backend image was
actually rebuilt (`docker compose up --build`), not just restarted.

**`pymysql.err.OperationalError: Access denied for user 'root'@'localhost'`**
This happens if MySQL's data volume was initialized with a different password in an
earlier run (MySQL only sets the root password on first initialization of an empty data
directory). Fix by clearing the volume so MySQL reinitializes fresh:
```
docker compose down -v
docker compose up --build
```

**Frontend can't reach the backend / API calls fail silently**
Check `frontend/vite.config.js` — its dev-server proxy must target the backend by its
Docker **service name** (`http://backend:5000`), not `127.0.0.1` or `localhost`, since
containers can't reach each other via loopback addresses.

## Cleaning up disk space (optional)

Docker images/containers accumulate over repeated builds. To reclaim space:
```
docker compose down
docker system prune -a --volumes
```
This removes all unused containers, images, and volumes. Safe to run — Docker rebuilds
whatever's needed next time.
