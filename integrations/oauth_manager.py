"""Google OAuth — click Connect, browser opens, done.

Credentials come from .env (GOOGLE_CLIENT_ID + GOOGLE_CLIENT_SECRET).
If not in .env, user enters them once in settings.
"""

import json
import os
import ssl
import threading
import subprocess
from logger import get_logger

log = get_logger("oauth")

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()

from config import DATA_DIR

REDIRECT_PORT = 8529
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/userinfo.email",
]


def _kc_get(a):
    try:
        r = subprocess.run(["security","find-generic-password","-s","Muse","-a",a,"-w"],
                           capture_output=True,text=True,timeout=5)
        return r.stdout.strip() if r.returncode==0 else ""
    except: return ""

def _kc_set(a, v):
    try:
        subprocess.run(["security","delete-generic-password","-s","Muse","-a",a],
                       capture_output=True,timeout=5)
        subprocess.run(["security","add-generic-password","-s","Muse","-a",a,"-w",v],
                       capture_output=True,timeout=5)
    except: pass

def _kc_delete(a):
    try: subprocess.run(["security","delete-generic-password","-s","Muse","-a",a],
                        capture_output=True,timeout=5)
    except: pass


class OAuthManager:
    def __init__(self):
        self._last_draft_id = None
        self._google_available = False
        # Check if Google libraries are installed
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.oauth2.credentials import Credentials
            self._google_available = True
        except ImportError:
            log.warning("google-auth-oauthlib not installed — Google integration disabled")
        # Load from env or keychain
        self.client_id = os.getenv("GOOGLE_CLIENT_ID","") or _kc_get("google_client_id")
        self.client_secret = os.getenv("GOOGLE_CLIENT_SECRET","") or _kc_get("google_client_secret")

    def save_credentials(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        _kc_set("google_client_id", client_id)
        _kc_set("google_client_secret", client_secret)

    def connect(self, emit_fn):
        """One click. Opens browser. User logs in. Done."""
        if not self._google_available:
            return {"ok": False, "error": "Google libraries not installed. Run: pip install google-api-python-client google-auth-oauthlib"}
        if not self.client_id or not self.client_secret:
            return {"ok": False, "needs_credentials": True}
        # Kill any existing OAuth server on the port
        try:
            subprocess.run(f"lsof -ti :{REDIRECT_PORT} | xargs kill -9",
                          shell=True, capture_output=True, timeout=3)
        except Exception:
            pass

        # Build config that google-auth-oauthlib expects
        config = {"installed": {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [f"http://localhost:{REDIRECT_PORT}"]
        }}
        tmp = str(DATA_DIR / "_tmp_oauth.json")
        with open(tmp, "w") as f:
            json.dump(config, f)

        def run():
            try:
                from google_auth_oauthlib.flow import InstalledAppFlow
                import urllib.request

                flow = InstalledAppFlow.from_client_secrets_file(tmp, SCOPES)
                creds = flow.run_local_server(
                    port=REDIRECT_PORT, prompt="consent",
                    access_type="offline", open_browser=True,
                )

                # Get email
                req = urllib.request.Request(
                    "https://www.googleapis.com/oauth2/v2/userinfo",
                    headers={"Authorization": f"Bearer {creds.token}"},
                )
                info = json.loads(urllib.request.urlopen(req, timeout=10, context=_SSL_CTX).read())
                email = info.get("email", "unknown")

                # Store token in keychain
                _kc_set(f"oauth:google:{email}", json.dumps({
                    "token": creds.token,
                    "refresh_token": creds.refresh_token,
                    "token_uri": creds.token_uri,
                    "client_id": creds.client_id,
                    "client_secret": creds.client_secret,
                    "scopes": list(creds.scopes or []),
                    "email": email,
                }))
                log.info("Connected: %s", email)
                emit_fn({"type": "oauth_complete", "service": "google", "email": email})
            except Exception as e:
                log.error("Auth failed: %s", e)
                emit_fn({"type": "oauth_error", "error": str(e)})
            finally:
                try: os.remove(tmp)
                except: pass

        threading.Thread(target=run, daemon=True).start()
        return {"ok": True}

    def get_token(self, service="google", email=""):
        if not email:
            accts = self.list_accounts(service)
            if not accts: return None
            email = accts[0]["email"]

        raw = _kc_get(f"oauth:{service}:{email}")
        if not raw: return None
        try: data = json.loads(raw)
        except: return None

        from google.oauth2.credentials import Credentials
        creds = Credentials(
            token=data.get("token"), refresh_token=data.get("refresh_token"),
            token_uri=data.get("token_uri","https://oauth2.googleapis.com/token"),
            client_id=data.get("client_id"), client_secret=data.get("client_secret"),
            scopes=data.get("scopes"),
        )
        if creds.expired and creds.refresh_token:
            try:
                from google.auth.transport.requests import Request
                creds.refresh(Request())
                data["token"] = creds.token
                _kc_set(f"oauth:{service}:{email}", json.dumps(data))
            except Exception as e:
                log.error("Refresh failed: %s — trying with existing token", e)

        return {"access_token": creds.token, "credentials": creds, "email": email}

    def remove_account(self, service, email):
        _kc_delete(f"oauth:{service}:{email}")

    def list_accounts(self, service=""):
        accounts = []
        try:
            r = subprocess.run(["security","dump-keychain"], capture_output=True, text=True, timeout=10)
            for line in r.stdout.split("\n"):
                if '"acct"' in line and "oauth:" in line:
                    for p in line.split('"'):
                        if p.startswith("oauth:"):
                            _, svc, em = p.split(":", 2)
                            if not service or svc == service:
                                accounts.append({"service": svc, "email": em})
        except: pass
        return accounts
