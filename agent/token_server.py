"""Mints a LiveKit token for each visitor of the LAN call page: a fresh room (one private call each) and identity.
Listens on 127.0.0.1 only; Caddy serves it over https at /token."""
import json
import os
import secrets
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from livekit import api

# livekit-server --dev defaults; set real keys for anything beyond local testing
KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] != "/token":
            self.send_error(404)
            return
        room, who = f"kayal-{secrets.token_hex(3)}", f"guest-{secrets.token_hex(3)}"
        token = (api.AccessToken(KEY, SECRET).with_identity(who).with_name("Guest").with_ttl(timedelta(hours=2))
                 .with_grants(api.VideoGrants(room_join=True, room=room)).to_jwt())
        body = json.dumps({"token": token, "room": room}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 18100), Handler).serve_forever()
