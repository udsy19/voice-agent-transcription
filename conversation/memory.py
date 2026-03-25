"""Agent memory: session history + long-term vector store.

Session: full conversation in RAM, passed to Claude every turn.
Long-term: LanceDB vector store, persisted to disk, survives restarts.
"""

import json
import time
from pathlib import Path
from logger import get_logger
from config import DATA_DIR

log = get_logger("memory")

MEMORY_DIR = DATA_DIR / "memory"
MAX_SESSION_TURNS = 40  # keep last N messages in session


class AgentMemory:
    def __init__(self):
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        self.session_history: list[dict] = []
        self._db = None
        self._table = None
        self._encoder = None
        self._init_vector_store()

    def _init_vector_store(self):
        """Initialize LanceDB for long-term memory."""
        try:
            import lancedb
            self._db = lancedb.connect(str(MEMORY_DIR / "lancedb"))
            # Create table if it doesn't exist
            try:
                self._table = self._db.open_table("memories")
                log.info("Loaded %d long-term memories", len(self._table))
            except Exception:
                # Table doesn't exist yet — will create on first insert
                self._table = None
                log.info("No existing memories, will create on first store")
        except Exception as e:
            log.warning("LanceDB init failed: %s, long-term memory disabled", e)

    def _get_encoder(self):
        """Lazy-load sentence transformer for embeddings."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
                log.info("Loaded embedding model")
            except Exception as e:
                log.warning("Sentence transformer failed: %s", e)
        return self._encoder

    def add_to_session(self, role: str, content: str):
        """Add a message to session history."""
        self.session_history.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })
        # Cap session length
        if len(self.session_history) > MAX_SESSION_TURNS:
            self.session_history = self.session_history[-MAX_SESSION_TURNS:]

    def get_session_messages(self) -> list[dict]:
        """Get session history formatted for Claude API."""
        return [{"role": m["role"], "content": m["content"]} for m in self.session_history]

    def retrieve_relevant(self, query: str, top_k: int = 8) -> list[str]:
        """Search long-term memory for relevant facts."""
        if not self._table or not self._db:
            return []
        encoder = self._get_encoder()
        if not encoder:
            return []

        try:
            embedding = encoder.encode(query).tolist()
            results = self._table.search(embedding).limit(top_k).to_list()
            return [r["text"] for r in results if "text" in r]
        except Exception as e:
            log.debug("Memory retrieval failed: %s", e)
            return []

    def store_facts(self, facts: list[str]):
        """Store extracted facts in long-term memory."""
        if not self._db or not facts:
            return
        encoder = self._get_encoder()
        if not encoder:
            return

        try:
            import pyarrow as pa

            embeddings = encoder.encode(facts).tolist()
            data = [
                {"text": fact, "vector": emb, "timestamp": time.time()}
                for fact, emb in zip(facts, embeddings)
            ]

            if self._table is None:
                self._table = self._db.create_table("memories", data)
            else:
                self._table.add(data)

            log.info("Stored %d facts in long-term memory", len(facts))
        except Exception as e:
            log.error("Failed to store facts: %s", e)

    def build_memory_block(self, query: str) -> str:
        """Build a memory context block for the system prompt."""
        memories = self.retrieve_relevant(query)
        if not memories:
            return ""
        return "\n".join(f"- {m}" for m in memories)

    async def extract_and_store(self, messages: list[dict], client=None):
        """Extract key facts from recent conversation and store them.

        Args:
            messages: Recent conversation messages
            client: anthropic.Anthropic client for extraction
        """
        if not client or not messages:
            return

        try:
            # Build conversation text
            convo_text = "\n".join(
                f"{m['role']}: {m['content']}" for m in messages
                if isinstance(m.get('content'), str)
            )

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": (
                        "Extract key facts, preferences, and entities from this conversation "
                        "worth remembering long-term. Return ONLY a JSON array of strings.\n"
                        "Example: [\"User's name is Udaya\", \"User works at Know Technologies\"]\n"
                        "If nothing worth remembering, return []\n\n"
                        f"Conversation:\n{convo_text}"
                    ),
                }],
            )

            text = response.content[0].text.strip()
            if text.startswith("["):
                facts = json.loads(text)
                if isinstance(facts, list) and facts:
                    self.store_facts([f for f in facts if isinstance(f, str)])
        except Exception as e:
            log.debug("Memory extraction failed: %s", e)
