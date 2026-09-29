#!/usr/bin/env python3
"""Local-only HTTP bridge for the interactive Trapdoor Prototype UI.

Run from the BHURAKSHAK repository with:
    .venv/bin/python scripts/trapdoor_inference_server.py

The service binds to loopback and reads only the frozen tabletop models.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.visualization.trapdoor_inference import TrapdoorInference  # noqa: E402

HOST = "127.0.0.1"
PORT = int(os.environ.get("BHURAKSHAK_SIM_PORT", "8765"))
inference = TrapdoorInference(ROOT)


class Handler(BaseHTTPRequestHandler):
    def _reply(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._reply(204, {})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._reply(200, {
                "ready": True,
                "model": "frozen tabletop Random Forest + Isolation Forest",
                "source": "generated stand-in dataset; not physical measurements",
                "window_samples": 20,
                "sample_hz": 10,
            })
            return
        self._reply(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/infer":
            self._reply(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2_000_000:
                raise ValueError("request body must be between 1 byte and 2 MB")
            payload = json.loads(self.rfile.read(length))
            samples = payload.get("samples")
            if not isinstance(samples, list) or len(samples) > 600:
                raise ValueError("samples must be a list with at most 600 entries")
            self._reply(200, inference.predict(samples))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._reply(400, {"error": str(exc)})

    def log_message(self, _format: str, *_args) -> None:
        return


if __name__ == "__main__":
    print(f"Trapdoor inference adapter ready at http://{HOST}:{PORT}")
    print("Inputs are simulated. Predictions are demonstrations, not safety alerts.")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()

