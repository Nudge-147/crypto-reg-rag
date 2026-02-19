#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_api_server.py
HTTP API wrapper for 03_retrieve_and_qa.py

Endpoints:
  GET  /health
  POST /query
"""

import importlib.util
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict


MODULE_PATH = Path(__file__).parent / "03_retrieve_and_qa.py"
SPEC = importlib.util.spec_from_file_location("retrieve_and_qa_03", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Failed to load module from {MODULE_PATH}")
MOD = importlib.util.module_from_spec(SPEC)
sys.modules["retrieve_and_qa_03"] = MOD
SPEC.loader.exec_module(MOD)  # type: ignore


def _json_response(handler: BaseHTTPRequestHandler, code: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class AppState:
    def __init__(self) -> None:
        api_key = os.getenv("GPTSAPI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing GPTSAPI_API_KEY")

        proxy_url = os.getenv("PROXY_URL", "")
        self.client, self.http_client = MOD.create_gptsapi_client(api_key, proxy_url)
        self.index, self.metas = MOD.load_index_and_meta(MOD.INDEX_PATH, MOD.META_PATH)
        self.authority_matrix = MOD.AuthorityMatrix()
        self.available_jurisdictions = MOD.get_available_jurisdictions(self.metas)

    def close(self) -> None:
        if self.http_client:
            self.http_client.close()


class QueryHandler(BaseHTTPRequestHandler):
    state: AppState = None  # type: ignore

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/health":
            _json_response(
                self,
                200,
                {
                    "ok": True,
                    "index_ntotal": int(self.state.index.ntotal),
                    "available_jurisdictions": self.state.available_jurisdictions,
                },
            )
            return

        if self.path == "/jurisdictions":
            _json_response(
                self,
                200,
                {
                    "available_jurisdictions": self.state.available_jurisdictions,
                    "count": len(self.state.available_jurisdictions),
                },
            )
            return

        else:
            _json_response(self, 404, {"error": "Not Found"})
            return

    def do_POST(self) -> None:
        if self.path != "/query":
            _json_response(self, 404, {"error": "Not Found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
        except Exception as e:
            _json_response(self, 400, {"error": f"Invalid JSON body: {e}"})
            return

        try:
            strict = os.getenv("VALIDATION_STRICT", "true").strip().lower() not in {
                "0",
                "false",
                "no",
            }
            request, validation = MOD.normalize_query_request(
                payload,
                self.state.available_jurisdictions,
                strict=strict,
            )
            response = MOD.execute_query(
                request,
                self.state.client,
                self.state.index,
                self.state.metas,
                self.state.authority_matrix,
            )
        except ValueError as e:
            _json_response(self, 400, {"error": str(e)})
            return
        except Exception as e:
            _json_response(self, 500, {"error": f"Query execution failed: {e}"})
            return

        items = []
        for r in response.retrieved_items:
            meta = r.get("chunk_meta", {})
            items.append(
                {
                    "final_score": r.get("final_score"),
                    "original_sim": r.get("original_sim"),
                    "chunk_id": meta.get("chunk_id"),
                    "doc_id": meta.get("doc_id"),
                    "jurisdiction": meta.get("jurisdiction"),
                    "text_preview": str(meta.get("text", ""))[:280],
                }
            )

        _json_response(
            self,
            200,
            {
                "request": {
                    "question": request.question,
                    "target_jurisdictions": request.target_jurisdictions,
                    "mode": request.mode,
                    "top_k": request.top_k,
                },
                "answer": response.answer,
                "applied_jurisdictions": response.applied_jurisdictions,
                "stats": response.stats,
                "warnings": response.warnings + validation.get("warnings", []),
                "validation": {
                    "strict": strict,
                    "invalid_jurisdictions": validation.get("invalid_jurisdictions", []),
                    "available_jurisdictions": validation.get("available_jurisdictions", []),
                },
                "retrieved_items": items,
            },
        )


def main() -> None:
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))

    state = AppState()
    QueryHandler.state = state
    server = ThreadingHTTPServer((host, port), QueryHandler)
    print(f"API server running at http://{host}:{port}")
    print("Endpoints: GET /health, GET /jurisdictions, POST /query")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        state.close()


if __name__ == "__main__":
    main()
