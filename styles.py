import os
import safe_json
from logger import get_logger

log = get_logger("styles")

from config import DATA_DIR
STYLES_PATH = str(DATA_DIR / "styles.json")

DEFAULT_STYLES = {
    "formal": {"description": "Professional and polished", "prompt": "Use formal tone: proper grammar, no contractions, professional language."},
    "casual": {"description": "Conversational and relaxed", "prompt": "Use casual tone: keep contractions, informal phrasing, conversational style."},
    "enthusiastic": {"description": "Energetic and positive", "prompt": "Use enthusiastic tone: positive energy, exclamation marks where natural, warm language."},
    "technical": {"description": "Precise and technical", "prompt": "Use technical tone: precise terminology, structured sentences, no filler."},
    "concise": {"description": "Brief and to the point", "prompt": "Be extremely concise: shortest possible phrasing, no unnecessary words."},
}

ROLE_DEFAULTS = {
    "developer": {"default_style": "technical", "app_overrides": {"Slack": "casual"}},
    "writer": {"default_style": "formal", "app_overrides": {"Slack": "casual", "Messages": "casual"}},
    "pm": {"default_style": "concise", "app_overrides": {"Slack": "casual", "Google Docs": "formal"}},
    "designer": {"default_style": "casual", "app_overrides": {"Google Docs": "formal"}},
    "sales": {"default_style": "enthusiastic", "app_overrides": {"Google Docs": "formal"}},
    "support": {"default_style": "casual", "app_overrides": {"Gmail": "formal", "Google Docs": "formal"}},
}


class StyleManager:
    def __init__(self):
        self._styles: dict = {}
        self._default_style: str = ""
        self._app_overrides: dict[str, str] = {}
        self._user_role: str = ""
        self._load()

    def _load(self):
        data = safe_json.load(STYLES_PATH, {})
        if isinstance(data, dict) and data:
            self._styles = data.get("styles") or DEFAULT_STYLES
            self._default_style = data.get("default_style") or ""
            self._app_overrides = data.get("app_overrides") or {}
            self._user_role = data.get("user_role") or ""
            log.info("Loaded: default=%s, role=%s, %d app overrides",
                     self._default_style, self._user_role, len(self._app_overrides))
        else:
            self._styles = DEFAULT_STYLES.copy()

    def _save(self):
        safe_json.save(STYLES_PATH, {
            "styles": self._styles,
            "default_style": self._default_style,
            "app_overrides": self._app_overrides,
            "user_role": self._user_role,
        })

    def setup_role(self, role: str):
        role = role.lower()
        if role in ROLE_DEFAULTS:
            defaults = ROLE_DEFAULTS[role]
            self._user_role = role
            self._default_style = defaults["default_style"]
            self._app_overrides = defaults["app_overrides"]
            self._save()
            log.info("Set role '%s', default style: %s", role, self._default_style)
        else:
            log.warning("Unknown role '%s'. Available: %s", role, ", ".join(ROLE_DEFAULTS.keys()))

    def get_style_prompt(self, app_name: str = "") -> str | None:
        style_name = None
        if app_name:
            for key, style in self._app_overrides.items():
                if key.lower() in app_name.lower():
                    style_name = style
                    break
        if not style_name:
            style_name = self._default_style
        if style_name and style_name in self._styles:
            return self._styles[style_name]["prompt"]
        return None

    def set_default(self, style_name: str):
        if style_name in self._styles:
            self._default_style = style_name
            self._save()
            log.info("Default style: %s", style_name)

    def set_app_override(self, app_name: str, style_name: str):
        if style_name in self._styles:
            self._app_overrides[app_name] = style_name
            self._save()
            log.info("%s -> %s", app_name, style_name)

    def add_style(self, name: str, description: str, prompt: str):
        self._styles[name] = {"description": description, "prompt": prompt}
        self._save()

    @property
    def is_setup(self) -> bool:
        return bool(self._user_role)

    def list_all(self) -> str:
        lines = [f"Role: {self._user_role or '(not set)'}", f"Default: {self._default_style or '(none)'}"]
        if self._app_overrides:
            lines.append("App overrides:")
            for app, style in self._app_overrides.items():
                lines.append(f"  {app} -> {style}")
        lines.append("Available styles:")
        for name, data in self._styles.items():
            lines.append(f"  {name}: {data['description']}")
        return "\n".join(lines)
