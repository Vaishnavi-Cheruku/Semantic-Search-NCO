# Semantic Search (Final)

This repository contains a semantic search project with a Python backend and a React frontend.

## Project Structure

- `backend/` - Python FastAPI backend (or Flask, depending on your code) and model inference logic.
- `frontend/` - React app created with Create React App.
- `data/` - Dataset CSV files used by the backend.
- `outputs/` - Generated embeddings, model files, and indices.
- `notebooks/` - Jupyter notebooks for experiments.

## Prerequisites

- Python 3.9+ (or 3.11 recommended)
- Node.js 18+ and npm
- A virtual environment for Python

## Setup and Run Backend

1. Open a terminal and navigate to the backend folder:

```bash
cd backend
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

4. Run the backend server:

```bash
uvicorn main:app --reload
```

5. The backend should run on `http://localhost:8000` and docs at `http://localhost:8000/docs`.

## Setup and Run Frontend

1. Open a new terminal and navigate to the frontend folder:

```bash
cd frontend
```

2. Install dependencies:

```bash
npm install
```

3. Start the frontend dev server:

```bash
npm start
```

4. Open the app at `http://localhost:3000`.

## Quick Start

1. Start backend server first.
2. Start frontend server.
3. Use the frontend UI to run semantic search queries.

## Notes

- Ensure `data/` and `outputs/` assets are present and paths in backend configs are correct.
- If API endpoints differ, update frontend API base URL in `frontend/src/App.js`.

## Troubleshooting

- If the frontend fails to fetch, check CORS in backend and base URL.
- If models or indices are missing, regenerate or download the required files from `outputs/`.
