"""Agent memory: session persistence + dialogue state + user profile + graph memory.

Session: full conversation in RAM, persisted to disk every 5 messages.
Long-term: LanceDB vector store for facts.
Graph: entity → relation → entity for deeper reasoning.
Profile: evolving user metadata.
Dialogue state: tracks current entity, last action, references.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from logger import get_logger
from config import DATA_DIR

log = get_logger("memory")

MEMORY_DIR = DATA_DIR / "memory"
SESSIONS_DIR = DATA_DIR / "sessions"
PROFILE_PATH = DATA_DIR / "user_profile.json"
GRAPH_PATH = DATA_DIR / "knowledge_graph.json"
MAX_SESSION_TURNS = 40


# ── User Profile ─────────────────────────────────────────────────────────────

@dataclass
class UserProfile:
    name: str = ""
    company: str = ""
    role: str = ""
    communication_style: str = "concise"  # concise, detailed
    technical_level: str = "intermediate"  # beginner, intermediate, expert
    preferences: dict = field(default_factory=dict)
    frequent_apps: list[str] = field(default_factory=list)
    frequent_contacts: list[str] = field(default_factory=list)
    last_updated: float = 0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "UserProfile":
        p = cls()
        for k, v in d.items():
            if hasattr(p, k):
                setattr(p, k, v)
        return p

    def to_prompt(self) -> str:
        parts = []
        if self.name:
            parts.append(f"User's name: {self.name}")
        if self.company:
            parts.append(f"Company: {self.company}")
        if self.role:
            parts.append(f"Role: {self.role}")
        if self.frequent_contacts:
            parts.append(f"Frequent contacts: {', '.join(self.frequent_contacts[:10])}")
        if self.preferences:
            parts.append(f"Preferences: {json.dumps(self.preferences)}")
        return "\n".join(parts) if parts else ""


# ── Dialogue State ───────────────────────────────────────────────────────────

@dataclass
class DialogueState:
    current_entity: str = ""  # "email from Satyam", "Linear ticket", etc.
    current_app: str = ""
    last_action: str = ""  # "opened email", "read calendar", etc.
    referenced_items: list[str] = field(default_factory=list)  # "that email", "this file"
    pending_confirmation: str = ""  # "send email to X" awaiting yes/no

    def resolve_reference(self, pronoun: str) -> str | None:
        """Resolve 'that', 'it', 'this one' to actual entity."""
        if pronoun.lower() in {"that", "it", "this", "that one", "this one"}:
            if self.referenced_items:
                return self.referenced_items[-1]
            if self.current_entity:
                return self.current_entity
        return None

    def update(self, transcript: str, action: str, entities: list[str] = None):
        self.last_action = action
        if entities:
            self.referenced_items = entities[-5:]  # keep last 5
            self.current_entity = entities[-1]

    def to_prompt(self) -> str:
        parts = []
        if self.current_entity:
            parts.append(f"Currently discussing: {self.current_entity}")
        if self.last_action:
            parts.append(f"Last action: {self.last_action}")
        if self.referenced_items:
            parts.append(f"'that'/'it' refers to: {self.referenced_items[-1]}")
        if self.pending_confirmation:
            parts.append(f"Awaiting confirmation: {self.pending_confirmation}")
        return "\n".join(parts) if parts else ""


# ── Knowledge Graph ──────────────────────────────────────────────────────────

class KnowledgeGraph:
    """Simple entity-relation-entity graph persisted to JSON."""

    def __init__(self):
        self._graph: dict[str, dict[str, list[str]]] = {}
        self._load()

    def _load(self):
        if GRAPH_PATH.exists():
            try:
                with open(GRAPH_PATH) as f:
                    self._graph = json.load(f)
                log.info("Loaded knowledge graph: %d entities", len(self._graph))
            except Exception:
                self._graph = {}

    def _save(self):
        try:
            with open(GRAPH_PATH, "w") as f:
                json.dump(self._graph, f, indent=2)
        except Exception as e:
            log.debug("Graph save failed: %s", e)

    def add(self, subject: str, relation: str, obj: str):
        if subject not in self._graph:
            self._graph[subject] = {}
        if relation not in self._graph[subject]:
            self._graph[subject][relation] = []
        if obj not in self._graph[subject][relation]:
            self._graph[subject][relation].append(obj)
            self._save()

    def query(self, entity: str, depth: int = 2) -> list[str]:
        """Get facts about an entity up to N hops deep."""
        facts = []
        visited = set()
        self._traverse(entity, depth, visited, facts)
        return facts

    def _traverse(self, entity: str, depth: int, visited: set, facts: list):
        if depth <= 0 or entity in visited:
            return
        visited.add(entity)
        if entity in self._graph:
            for rel, targets in self._graph[entity].items():
                for t in targets:
                    facts.append(f"{entity} {rel} {t}")
                    self._traverse(t, depth - 1, visited, facts)

    def to_prompt(self, query: str) -> str:
        """Get relevant graph facts for a query."""
        words = query.lower().split()
        facts = []
        for entity in self._graph:
            if any(w in entity.lower() for w in words):
                facts.extend(self.query(entity, depth=2))
        return "\n".join(f"- {f}" for f in facts[:10]) if facts else ""


# ── Main Memory Class ────────────────────────────────────────────────────────

class AgentMemory:
    def __init__(self):
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

        self.session_history: list[dict] = []
        self.dialogue: DialogueState = DialogueState()
        self.profile: UserProfile = self._load_profile()
        self.graph: KnowledgeGraph = KnowledgeGraph()
        self._session_id = str(int(time.time()))
        self._db = None
        self._table = None
        self._encoder = None

        self._load_last_session()
        self._init_vector_store()

    # ── Session persistence ──────────────────────────────────────────────────

    def _load_last_session(self):
        sessions = sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stem)
        if not sessions:
            return
        try:
            with open(sessions[-1]) as f:
                data = json.load(f)
            self.session_history = data.get("history", [])[-MAX_SESSION_TURNS:]
            ds = data.get("dialogue", {})
            self.dialogue = DialogueState(**{k: v for k, v in ds.items() if hasattr(self.dialogue, k)})
            self._session_id = sessions[-1].stem
            log.info("Restored session %s (%d messages)", self._session_id, len(self.session_history))
        except Exception as e:
            log.warning("Session restore failed: %s", e)

    def save_session(self):
        try:
            path = SESSIONS_DIR / f"{self._session_id}.json"
            data = {
                "history": self.session_history[-MAX_SESSION_TURNS:],
                "dialogue": self.dialogue.__dict__,
                "timestamp": time.time(),
            }
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            log.debug("Session save failed: %s", e)

    def new_session(self):
        self._session_id = str(int(time.time()))
        self.session_history = []
        self.dialogue = DialogueState()

    # ── Profile persistence ──────────────────────────────────────────────────

    def _load_profile(self) -> UserProfile:
        if PROFILE_PATH.exists():
            try:
                with open(PROFILE_PATH) as f:
                    return UserProfile.from_dict(json.load(f))
            except Exception:
                pass
        return UserProfile()

    def save_profile(self):
        try:
            self.profile.last_updated = time.time()
            with open(PROFILE_PATH, "w") as f:
                json.dump(self.profile.to_dict(), f, indent=2)
        except Exception as e:
            log.debug("Profile save failed: %s", e)

    # ── Session operations ───────────────────────────────────────────────────

    def add_to_session(self, role: str, content: str):
        self.session_history.append({
            "role": role, "content": content, "timestamp": time.time(),
        })
        if len(self.session_history) > MAX_SESSION_TURNS:
            self.session_history = self.session_history[-MAX_SESSION_TURNS:]
        # Auto-save every 5 messages
        if len(self.session_history) % 5 == 0:
            self.save_session()

    def get_session_messages(self) -> list[dict]:
        return [{"role": m["role"], "content": m["content"]} for m in self.session_history]

    # ── Vector store ─────────────────────────────────────────────────────────

    def _init_vector_store(self):
        try:
            import lancedb
            self._db = lancedb.connect(str(MEMORY_DIR / "lancedb"))
            try:
                self._table = self._db.open_table("memories")
                log.info("Loaded %d long-term memories", len(self._table))
            except Exception:
                self._table = None
                log.info("No existing memories")
        except Exception as e:
            log.warning("LanceDB init failed: %s", e)

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                log.warning("Encoder failed: %s", e)
        return self._encoder

    def retrieve_relevant(self, query: str, top_k: int = 8) -> list[str]:
        if not self._table or not self._db:
            return []
        encoder = self._get_encoder()
        if not encoder:
            return []
        try:
            embedding = encoder.encode(query).tolist()
            results = self._table.search(embedding).limit(top_k).to_list()
            return [r["text"] for r in results if "text" in r]
        except Exception:
            return []

    def store_facts(self, facts: list[str]):
        if not self._db or not facts:
            return
        encoder = self._get_encoder()
        if not encoder:
            return
        try:
            embeddings = encoder.encode(facts).tolist()
            data = [{"text": f, "vector": e, "timestamp": time.time()} for f, e in zip(facts, embeddings)]
            if self._table is None:
                self._table = self._db.create_table("memories", data)
            else:
                self._table.add(data)
            log.info("Stored %d facts", len(facts))
        except Exception as e:
            log.error("Store facts failed: %s", e)

    # ── Context building ─────────────────────────────────────────────────────

    def build_memory_block(self, query: str) -> str:
        """Build full memory context for system prompt."""
        parts = []

        # User profile
        profile_str = self.profile.to_prompt()
        if profile_str:
            parts.append(f"USER PROFILE:\n{profile_str}")

        # Dialogue state
        dialogue_str = self.dialogue.to_prompt()
        if dialogue_str:
            parts.append(f"CURRENT CONTEXT:\n{dialogue_str}")

        # Graph facts
        graph_str = self.graph.to_prompt(query)
        if graph_str:
            parts.append(f"KNOWN FACTS:\n{graph_str}")

        # Vector memory
        memories = self.retrieve_relevant(query)
        if memories:
            parts.append("LONG-TERM MEMORY:\n" + "\n".join(f"- {m}" for m in memories))

        return "\n\n".join(parts)

    # ── Memory extraction ────────────────────────────────────────────────────

    async def extract_and_store(self, messages: list[dict], client=None):
        """Extract facts, update profile, update graph from recent conversation."""
        if not client or not messages:
            return

        convo = "\n".join(f"{m['role']}: {m['content']}" for m in messages if isinstance(m.get('content'), str))

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": (
                    "Extract from this conversation:\n\n"
                    "1. Key facts (JSON array of strings)\n"
                    "2. User profile updates (JSON object: name, company, role, preferences)\n"
                    "3. Relationships (JSON array of [subject, relation, object])\n\n"
                    "Return JSON:\n"
                    '{"facts": [...], "profile": {...}, "relations": [[s,r,o], ...]}\n\n'
                    f"Conversation:\n{convo}"
                )}],
            )

            data = json.loads(response.content[0].text.strip())

            # Store facts
            facts = data.get("facts", [])
            if facts:
                self.store_facts([f for f in facts if isinstance(f, str)])

            # Update profile
            profile_updates = data.get("profile", {})
            if profile_updates:
                for k, v in profile_updates.items():
                    if v and hasattr(self.profile, k):
                        setattr(self.profile, k, v)
                self.save_profile()

            # Update graph
            relations = data.get("relations", [])
            for rel in relations:
                if isinstance(rel, list) and len(rel) == 3:
                    self.graph.add(rel[0], rel[1], rel[2])

        except Exception as e:
            log.debug("Memory extraction failed: %s", e)
