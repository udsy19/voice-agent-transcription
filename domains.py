"""Domain-specific terminology modes.

Each mode injects specialized vocabulary into both Whisper (initial_prompt)
and Groq (system prompt context) for dramatically better accuracy on
domain-specific terms.
"""

import json
import os
from logger import get_logger

log = get_logger("domains")

from config import DATA_DIR
DOMAINS_PATH = str(DATA_DIR / "domains.json")

BUILTIN_DOMAINS = {
    "tech": {
        "description": "Software engineering & tech",
        "terms": [
            "API", "REST", "GraphQL", "SQL", "NoSQL", "PostgreSQL", "MongoDB",
            "Redis", "Docker", "Kubernetes", "AWS", "GCP", "Azure", "Vercel",
            "Supabase", "Firebase", "OAuth", "JWT", "CORS", "CDN", "CI/CD",
            "TypeScript", "JavaScript", "Python", "React", "Next.js", "Node.js",
            "npm", "pip", "git", "GitHub", "VS Code", "Cursor", "Webpack",
            "localhost", "backend", "frontend", "middleware", "microservice",
            "repository", "pull request", "merge conflict", "deployment",
            "refactor", "debugging", "stack trace", "runtime", "compile",
            "async", "await", "callback", "promise", "webhook", "endpoint",
        ],
        "prompt_hint": "The user is a software developer. Preserve technical terms exactly as spoken. Use camelCase/snake_case when contextually appropriate.",
    },
    "medical": {
        "description": "Healthcare & medical",
        "terms": [
            "diagnosis", "prognosis", "prescription", "contraindication",
            "pathology", "radiology", "cardiology", "neurology", "oncology",
            "hematology", "immunology", "pharmacology", "epidemiology",
            "MRI", "CT scan", "ECG", "EKG", "ICU", "ER", "OR",
            "hypertension", "diabetes", "cholesterol", "anesthesia",
            "biopsy", "chemotherapy", "dialysis", "intubation",
            "HIPAA", "EMR", "EHR", "ICD-10", "CPT",
        ],
        "prompt_hint": "The user is a medical professional. Spell medical terms and drug names correctly. Use proper medical terminology.",
    },
    "legal": {
        "description": "Legal & compliance",
        "terms": [
            "plaintiff", "defendant", "arbitration", "litigation",
            "deposition", "subpoena", "affidavit", "injunction",
            "jurisdiction", "statute", "precedent", "tort",
            "indemnification", "liability", "compliance", "regulatory",
            "due diligence", "fiduciary", "escrow", "probate",
            "GDPR", "SOC 2", "CCPA", "SEC", "FTC",
        ],
        "prompt_hint": "The user is a legal professional. Spell legal terms correctly. Maintain formal language.",
    },
    "finance": {
        "description": "Finance & investing",
        "terms": [
            "P&L", "EBITDA", "ROI", "IRR", "NPV", "CAGR", "AUM",
            "IPO", "M&A", "LBO", "DCF", "PE ratio", "market cap",
            "venture capital", "private equity", "hedge fund", "portfolio",
            "derivative", "equity", "bond", "yield", "dividend",
            "SEC filing", "10-K", "10-Q", "S-1", "cap table",
        ],
        "prompt_hint": "The user works in finance. Spell financial terms and acronyms correctly.",
    },
}


class DomainManager:
    def __init__(self):
        self._domains: dict = {}
        self._active: str = ""
        self._load()

    def _load(self):
        self._domains = BUILTIN_DOMAINS.copy()
        if os.path.exists(DOMAINS_PATH):
            try:
                with open(DOMAINS_PATH) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    custom = data.get("custom_domains") or {}
                    self._domains.update(custom)
                    self._active = data.get("active") or ""
                log.info("Loaded domains, active=%s", self._active or "(none)")
            except (json.JSONDecodeError, IOError) as e:
                log.error("Failed to load domains: %s", e)

    def _save(self):
        try:
            # Only save custom domains and active state
            custom = {k: v for k, v in self._domains.items() if k not in BUILTIN_DOMAINS}
            with open(DOMAINS_PATH, "w") as f:
                json.dump({"custom_domains": custom, "active": self._active}, f, indent=2)
        except IOError as e:
            log.error("Failed to save domains: %s", e)

    def set_active(self, domain: str):
        if domain and domain not in self._domains:
            log.warning("Unknown domain: %s", domain)
            return
        self._active = domain
        self._save()
        log.info("Active domain: %s", domain or "(none)")

    def get_active(self) -> str:
        return self._active

    def get_whisper_prompt(self) -> str | None:
        """Get domain terms as Whisper initial_prompt."""
        if not self._active or self._active not in self._domains:
            return None
        terms = self._domains[self._active].get("terms", [])
        return ", ".join(terms[:40]) if terms else None

    def get_cleaner_hint(self) -> str | None:
        """Get domain-specific hint for the Groq cleaner."""
        if not self._active or self._active not in self._domains:
            return None
        return self._domains[self._active].get("prompt_hint")

    def list_domains(self) -> dict:
        return {k: v.get("description", "") for k, v in self._domains.items()}

    def add_domain(self, name: str, description: str, terms: list[str], prompt_hint: str = ""):
        self._domains[name] = {
            "description": description,
            "terms": terms,
            "prompt_hint": prompt_hint,
        }
        self._save()
        log.info("Added domain: %s (%d terms)", name, len(terms))
