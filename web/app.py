from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.simulation_api import SimulateRequest, run_from_request, scene_presets


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="MPC Interactive Sandbox")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/presets")
def presets():
    return {"scenes": scene_presets()}


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    try:
        return run_from_request(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
