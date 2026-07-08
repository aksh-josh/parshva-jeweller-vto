# AI Powered Virtual Try On

ProMatch is a web app that finds the professional footballers who are most similar to you.
You set six attributes (pace, shooting, passing, dribbling, defending, physical), pick how you
want to match, and it shows the players whose stats are the closest, with a match percentage.

## Why I made this

I play sport, and one question I always found interesting is "which professional do I actually
play like". Most people answer that from gut feeling. I wanted to answer it with numbers. So
ProMatch takes your six attributes and compares them against around 17,000 real players to find
who you are closest to.

A few things I like about it:
- it does not just sort by overall rating, it compares the actual shape of your stats
- you can match in two ways: by overall similarity, or by playing style
- it runs the comparison against the whole dataset, not a small sample

## How it works

There are three parts, each in its own Docker container:
- frontend: a React page with the form and the results
- backend: a Python (FastAPI) API that does the matching
- database: PostgreSQL, which stores the players

The frontend never talks to the database directly. It calls the backend API, and the backend
reads from the database. The three containers are connected over a custom Docker network.

For the matching I use k-nearest-neighbours from scikit-learn. Each player is a point made of
their six attribute values. When you send your attributes, the backend finds the players nearest
to you. "Nearest" is measured in one of two ways:
- Euclidean distance, which is overall similarity (similar numbers)
- Cosine, which is playing style (similar shape, even if the overall level is different)

## How to run it

You need Docker installed. From the project folder run:

```
docker compose up
```

Then open:
- the app: http://localhost:5173
- the backend health check: http://localhost:8000/health

The database loads the player data by itself the first time the backend starts, so there is
nothing else to set up.

If you want to check the data is in the database:

```
docker compose exec db psql -U promatch -d promatch -c "SELECT COUNT(*) FROM players;"
```

## How to use it

1. Move the six sliders to your attributes.
2. Choose "Overall similarity" or "Playing style".
3. Click "Find matches".
4. You get the closest players, each with a match percentage (higher means more similar).

## The data

I used the FIFA 22 complete player dataset. The original is on Kaggle (by Stefano Leone). I
downloaded the players_22.csv file from a public copy of it on GitHub:
https://github.com/abineshta/FIFA-22-complete-player-dataset-EDA

The raw file is large and has 110 columns I do not need, so I do not keep it in the repo. The
cleaned file (data/players.csv) is in the repo and that is what the app loads.

## How I prepared the data

I cleaned the raw data in a notebook (data/data_cleaning.ipynb). The steps were:
- keep only the columns I need (name, club, nationality, age, overall, positions, and the six attributes)
- rename a few columns to simpler names (for example physic to physical)
- remove goalkeepers, because their pace/shooting/etc are empty (they are rated differently)
- drop any rows that still had missing values
- save the result as data/players.csv

I also worked out the matching logic in a second notebook (data/similarity.ipynb) before I put
it into the backend, so I could test it on the data first.

## Tools I used

- Python and FastAPI for the backend and API
- scikit-learn for the k-nearest-neighbours matching
- PostgreSQL for the database
- React (with Vite) for the frontend
- pandas for cleaning the data
- Docker and docker compose to run everything together

## Project structure

```
ProMatch/
  docker-compose.yml        starts the three containers on a custom network
  backend/
    main.py                 the API (health, players, meta, recommend)
    recommender.py          the matching logic (k-NN)
    database.py             connects to the database
    seed_data.py            loads players.csv into the database on startup
    Dockerfile
    requirements.txt
  frontend/
    src/
      App.jsx               the form and the results
      api.js                the calls to the backend
      styles.css
    index.html
    Dockerfile
    package.json
  data/
    players.csv             the cleaned player data
    data_cleaning.ipynb     how I cleaned the data
    similarity.ipynb        how I built the matching
```
