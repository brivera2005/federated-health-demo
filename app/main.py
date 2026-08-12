from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.federated import session
from app.models import DemoState

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"

app = FastAPI(
 title="Federated Health Demo",
 description="Two-hospital federated readmission risk training - weights only leave site (demo / portfolio).",
 version="1.0.0",
)


@app.get("/api/health")
def health():
 return {"status": "ok", "step": session.state.step}


@app.get("/api/state", response_model=DemoState)
def get_state():
 return session.state


@app.post("/api/step", response_model=DemoState)
def advance_step():
 return session.advance()


@app.post("/api/reset", response_model=DemoState)
def reset():
 return session.reset()


@app.get("/")
def index():
 return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
