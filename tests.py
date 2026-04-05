#!/usr/bin/env python3
"""Muse E2E Test Suite — simulates voice commands and validates responses.

Run: python3 tests.py
Requires: backend running on localhost:8528

Tests send text directly to the assistant (bypassing mic/Whisper)
and validate that the correct tools are called and responses make sense.
"""

import sys
import os
import time
import json
import requests

sys.path.insert(0, os.path.dirname(__file__))

API = "http://127.0.0.1:8528"
PASS = 0
FAIL = 0
SKIP = 0


def check_health():
    try:
        r = requests.get(f"{API}/api/health", timeout=5)
        return r.json().get("ok", False)
    except Exception:
        return False


def send_to_assistant(text: str) -> dict:
    """Send text directly to the assistant (simulating voice input)."""
    # Import and call assistant directly
    from assistant import Assistant
    from integrations.oauth_manager import OAuthManager
    from todos import TodoList
    from brain import Brain
    from config import GROQ_API_KEY
    from groq import Groq

    if not hasattr(send_to_assistant, '_assistant'):
        client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
        oauth = OAuthManager()
        todos = TodoList()
        brain = Brain()
        send_to_assistant._assistant = Assistant(client, oauth, lambda msg: None, todos, brain)

    a = send_to_assistant._assistant
    try:
        response = a.handle(text)
        return {"ok": True, "response": response or ""}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def test(name: str, command: str, expect_contains: list[str] = None,
         expect_not_contains: list[str] = None, expect_ok: bool = True,
         tool_name: str = None, timeout_sec: int = 30):
    """Run a single test case."""
    global PASS, FAIL, SKIP

    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  Command: \"{command}\"")

    try:
        start = time.time()
        result = send_to_assistant(command)
        duration = time.time() - start

        response = result.get("response", result.get("error", ""))
        ok = result.get("ok", False)

        print(f"  Duration: {duration:.1f}s")
        print(f"  Response: {response[:120]}")

        passed = True

        if expect_ok and not ok:
            print(f"  FAIL: Expected ok=True but got error")
            passed = False

        if expect_contains:
            for term in expect_contains:
                if term.lower() not in response.lower():
                    print(f"  FAIL: Expected response to contain '{term}'")
                    passed = False

        if expect_not_contains:
            for term in expect_not_contains:
                if term.lower() in response.lower():
                    print(f"  FAIL: Response should NOT contain '{term}'")
                    passed = False

        if duration > timeout_sec:
            print(f"  FAIL: Took {duration:.1f}s (limit: {timeout_sec}s)")
            passed = False

        if passed:
            print(f"  PASS ✓")
            PASS += 1
        else:
            FAIL += 1

    except Exception as e:
        print(f"  ERROR: {e}")
        FAIL += 1


def test_api(name: str, method: str, path: str, body: dict = None,
             expect_key: str = None, expect_value=None):
    """Test an API endpoint directly."""
    global PASS, FAIL

    print(f"\n{'='*60}")
    print(f"API TEST: {name}")

    try:
        if method == "GET":
            r = requests.get(f"{API}{path}", timeout=10)
        else:
            r = requests.post(f"{API}{path}", json=body or {}, timeout=10)

        data = r.json()
        print(f"  Status: {r.status_code}")

        if r.status_code != 200:
            print(f"  FAIL: HTTP {r.status_code}")
            FAIL += 1
            return

        if expect_key:
            val = data.get(expect_key)
            if expect_value is not None:
                if val == expect_value:
                    print(f"  PASS ✓ ({expect_key}={val})")
                    PASS += 1
                else:
                    print(f"  FAIL: {expect_key}={val}, expected {expect_value}")
                    FAIL += 1
            elif val is not None:
                print(f"  PASS ✓ ({expect_key} exists)")
                PASS += 1
            else:
                print(f"  FAIL: {expect_key} missing")
                FAIL += 1
        else:
            print(f"  PASS ✓")
            PASS += 1

    except Exception as e:
        print(f"  ERROR: {e}")
        FAIL += 1


def run_all():
    global PASS, FAIL, SKIP

    print("\n" + "=" * 60)
    print("MUSE E2E TEST SUITE")
    print("=" * 60)

    # ── API Health ──
    test_api("Health check", "GET", "/api/health", expect_key="ok", expect_value=True)
    test_api("State", "GET", "/api/state", expect_key="status")
    test_api("Voices", "GET", "/api/voices", expect_key="voices")
    test_api("Todos", "GET", "/api/todos", expect_key="todos")
    test_api("Memories", "GET", "/api/memories", expect_key="memories")
    test_api("Privacy", "GET", "/api/privacy", expect_key="clipboard_context")

    # ── Calendar ──
    test("Calendar today",
         "What's on my calendar today?",
         timeout_sec=15)

    test("Calendar this week",
         "What does my week look like?",
         timeout_sec=15)

    # ── Todos ──
    test("Add todo",
         "Add buy milk to my todo list",
         expect_contains=["milk"],
         timeout_sec=20)

    test("List todos",
         "What's on my todo list?",
         timeout_sec=15)

    test("Complete todo",
         "I bought the milk, check it off",
         expect_contains=["milk"],
         timeout_sec=20)

    # ── Memory ──
    test("Remember fact",
         "Remember that my favorite color is blue",
         expect_contains=["blue"],
         timeout_sec=25)

    test("Recall fact",
         "What's my favorite color?",
         expect_contains=["blue"],
         timeout_sec=15)

    test("Forget fact",
         "Forget my favorite color",
         timeout_sec=25)

    # ── Clipboard ──
    # Set clipboard first
    import subprocess
    subprocess.run(["pbcopy"], input="def hello(): print('world')", text=True)
    test("Explain clipboard",
         "Explain this code",
         timeout_sec=20)

    # ── System control ──
    test("Set volume",
         "Set volume to 30",
         timeout_sec=20)

    # ── Screen analysis ──
    test("Analyze screen",
         "What am I looking at on my screen?",
         timeout_sec=30)

    # ── Weekly reflection ──
    test("Weekly reflection",
         "How was my week?",
         timeout_sec=15)

    # ── iMessage ──
    test("Check messages",
         "Who was my last text from?",
         timeout_sec=25)

    # ── Standup ──
    test_api("Standup via API", "GET", "/api/briefing", expect_key="text")

    # ── Edge cases ──
    test("Empty/vague command",
         "Hello",
         timeout_sec=20)

    test("Nonsense input",
         "asdfghjkl qwerty",
         timeout_sec=20)

    test("Very long input",
         "Can you please help me with something? " * 10,
         timeout_sec=25)

    # ── Rate limit resilience ──
    test("Rapid fire 1",
         "What time is it?",
         timeout_sec=15)

    test("Rapid fire 2",
         "What day is it?",
         timeout_sec=20)

    # ── Results ──
    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    print(f"Total: {PASS + FAIL + SKIP} tests")
    print("=" * 60)

    return FAIL == 0


if __name__ == "__main__":
    if not check_health():
        print("ERROR: Backend not running. Start with ./start.sh first.")
        sys.exit(1)

    success = run_all()
    sys.exit(0 if success else 1)
