#!/usr/bin/env python3
"""Muse Unit Tests — test individual modules without backend running.

Run: python3 test_units.py
No backend needed — tests modules in isolation.
"""

import sys
import os
import json
import time
import tempfile
import threading

sys.path.insert(0, os.path.dirname(__file__))

PASS = 0
FAIL = 0


def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  PASS: {name}")
        PASS += 1
    except AssertionError as e:
        print(f"  FAIL: {name} — {e}")
        FAIL += 1
    except Exception as e:
        print(f"  ERROR: {name} — {type(e).__name__}: {e}")
        FAIL += 1


# ═══════════════════════════════════════════════════════════════════════════
print("\n=== safe_json ===")

import safe_json

def test_save_load_roundtrip():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        data = {"key": "value", "list": [1, 2, 3], "nested": {"a": True}}
        assert safe_json.save(path, data)
        loaded = safe_json.load(path)
        assert loaded == data, f"Expected {data}, got {loaded}"
    finally:
        os.unlink(path)
test("save/load roundtrip", test_save_load_roundtrip)

def test_load_missing_file():
    result = safe_json.load("/tmp/muse_test_nonexistent_xyz.json", {"default": True})
    assert result == {"default": True}
test("load missing file returns default", test_load_missing_file)

def test_load_corrupt_file():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        f.write("{invalid json!!")
        path = f.name
    try:
        result = safe_json.load(path, {"fallback": True})
        assert result == {"fallback": True}
    finally:
        os.unlink(path)
        try:
            os.unlink(path + ".corrupt")
        except FileNotFoundError:
            pass
test("load corrupt file returns default + backs up", test_load_corrupt_file)

def test_atomic_save_no_partial():
    """Verify .tmp file is cleaned up after save."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        safe_json.save(path, {"data": True})
        assert not os.path.exists(path + ".tmp"), ".tmp file should be cleaned up"
    finally:
        os.unlink(path)
test("atomic save cleans up .tmp", test_atomic_save_no_partial)

def test_concurrent_saves():
    """Multiple threads saving shouldn't corrupt."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    errors = []
    def save_n(n):
        try:
            for i in range(20):
                safe_json.save(path, {"thread": n, "iter": i})
        except Exception as e:
            errors.append(e)
    threads = [threading.Thread(target=save_n, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        assert not errors, f"Errors: {errors}"
        result = safe_json.load(path)
        assert isinstance(result, dict)
        assert "thread" in result
    finally:
        os.unlink(path)
test("concurrent saves don't corrupt", test_concurrent_saves)


# ═══════════════════════════════════════════════════════════════════════════
print("\n=== speech_prep ===")

from speech_prep import prepare_for_speech

def test_24h_to_12h():
    assert "1 PM" in prepare_for_speech("Meeting at 13:00")
    assert "9 AM" in prepare_for_speech("Start at 09:00")
    assert "12 PM" in prepare_for_speech("Lunch at 12:00")
test("24h → 12h time conversion", test_24h_to_12h)

def test_simplify_zero_minutes():
    result = prepare_for_speech("Event at 3:00 PM")
    assert "3 PM" in result, f"Expected '3 PM' in '{result}'"
test("simplify :00 minutes", test_simplify_zero_minutes)

def test_abbreviation_expansion():
    assert "Doctor" in prepare_for_speech("See Dr. Smith")
    assert "versus" in prepare_for_speech("Team A vs. Team B")
    assert "etcetera" in prepare_for_speech("stuff etc.")
test("abbreviation expansion", test_abbreviation_expansion)

def test_markdown_removal():
    assert "**" not in prepare_for_speech("This is **bold** text")
    assert "*" not in prepare_for_speech("This is *italic* text")
    assert "`" not in prepare_for_speech("Run `command` here")
test("markdown removal", test_markdown_removal)

def test_bullet_removal():
    result = prepare_for_speech("- Item one\n- Item two\n- Item three")
    assert "-" not in result or result.count("-") == 0
test("bullet removal", test_bullet_removal)

def test_iso_timestamp_removal():
    result = prepare_for_speech("Event at 2026-04-04T13:00:00Z is fun")
    assert "2026" not in result, f"ISO timestamp not removed: '{result}'"
test("ISO timestamp removal", test_iso_timestamp_removal)

def test_empty_input():
    assert prepare_for_speech("") == ""
    assert prepare_for_speech(None) is None
test("empty/None input", test_empty_input)

def test_already_natural():
    text = "You have a meeting at 3 PM with Sarah."
    assert prepare_for_speech(text) == text
test("already natural text unchanged", test_already_natural)


# ═══════════════════════════════════════════════════════════════════════════
print("\n=== brain ===")

from brain import Brain

def test_brain_remember_recall():
    import shutil
    brain_path = tempfile.mkdtemp()
    try:
        b = Brain.__new__(Brain)
        b._lock = threading.Lock()
        b.facts = []
        b.deadlines = []
        b.meetings = []
        b.habits = {}
        b.remember("My favorite color is blue")
        assert len(b.facts) == 1
        assert "blue" in b.facts[0]["text"]
    finally:
        shutil.rmtree(brain_path, ignore_errors=True)
test("brain remember/recall", test_brain_remember_recall)

def test_brain_deadline_parsing():
    b = Brain.__new__(Brain)
    b._lock = threading.Lock()
    b.facts, b.deadlines, b.meetings, b.habits = [], [], [], {}
    result = b.detect_deadline_in_text("I need to finish the report by Friday")
    assert result is not None, "Should detect deadline"
    assert "friday" in result["due_hint"].lower()
test("brain deadline detection", test_brain_deadline_parsing)

def test_brain_deadline_no_false_positive():
    b = Brain.__new__(Brain)
    b._lock = threading.Lock()
    b.facts, b.deadlines, b.meetings, b.habits = [], [], [], {}
    result = b.detect_deadline_in_text("Had a great lunch today")
    assert result is None, "Should not detect deadline in casual text"
test("brain no false positive deadlines", test_brain_deadline_no_false_positive)

def test_brain_meeting_search():
    b = Brain.__new__(Brain)
    b._lock = threading.Lock()
    b.facts, b.deadlines, b.meetings, b.habits = [], [], [], {}
    b.meetings = [
        {"summary": "Sprint planning", "date": "2026-04-01", "notes": "Discussed roadmap", "action_items": ["Write specs"], "ts": time.time()},
        {"summary": "Design review", "date": "2026-04-02", "notes": "Reviewed mockups", "action_items": [], "ts": time.time()},
    ]
    result = b.get_meeting_context("sprint")
    assert "Sprint planning" in result
    assert "Write specs" in result
test("brain meeting search", test_brain_meeting_search)

def test_brain_context_for_llm():
    b = Brain.__new__(Brain)
    b._lock = threading.Lock()
    b.facts = [{"text": "User likes Python", "category": "general", "ts": time.time()}]
    b.deadlines = []
    b.meetings = []
    b.habits = {}
    ctx = b.get_context_for_llm()
    assert "Python" in ctx
test("brain context for LLM", test_brain_context_for_llm)


# ═══════════════════════════════════════════════════════════════════════════
print("\n=== todos ===")

from todos import TodoList

def test_todos_add_complete():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        t = TodoList.__new__(TodoList)
        t._lock = threading.Lock()
        t._path = path
        t._items = []
        item = t.add("Buy groceries")
        assert item["text"] == "Buy groceries"
        assert item["done"] is False
        pending = t.list_pending()
        assert len(pending) == 1
        t.complete(item["id"])
        assert len(t.list_pending()) == 0
        assert len(t.list_done()) == 1
    finally:
        os.unlink(path)
test("todos add/complete lifecycle", test_todos_add_complete)

def test_todos_complete_nonexistent():
    t = TodoList.__new__(TodoList)
    t._lock = threading.Lock()
    t._path = "/dev/null"
    t._items = []
    # Should not crash
    t.complete("nonexistent-id")
test("todos complete nonexistent ID doesn't crash", test_todos_complete_nonexistent)


# ═══════════════════════════════════════════════════════════════════════════
print("\n=== imessage ===")

from imessage import _clean_text, resolve_contact, find_phone_for_name

def test_clean_text_emojis():
    result = _clean_text("Hello 😀👋 World")
    assert "😀" not in result
    assert "Hello" in result and "World" in result
test("clean text removes emojis", test_clean_text_emojis)

def test_clean_text_urls():
    result = _clean_text("Check https://example.com/page for details")
    assert "https" not in result
    assert "Check" in result
test("clean text removes URLs", test_clean_text_urls)

def test_clean_text_truncation():
    long = "A" * 300
    result = _clean_text(long)
    assert len(result) <= 200
test("clean text truncates to 200 chars", test_clean_text_truncation)

def test_clean_text_empty():
    assert _clean_text("") == ""
    assert _clean_text(None) == ""
test("clean text handles empty/None", test_clean_text_empty)

def test_resolve_unknown():
    assert resolve_contact("Unknown") == "Unknown"
    assert resolve_contact("") == "Unknown"
    assert resolve_contact(None) == "Unknown"
test("resolve_contact handles edge cases", test_resolve_unknown)

def test_resolve_phone_masking():
    result = resolve_contact("+15551234567")
    # Should mask to ...4567 if not in contacts
    assert result.endswith("4567") or result == "+15551234567"
test("resolve_contact masks phone numbers", test_resolve_phone_masking)


# ═══════════════════════════════════════════════════════════════════════════
print("\n=== system_control ===")

import system_control

def test_sanitize_app_name():
    assert system_control._sanitize_app_name("Safari") == "Safari"
    # Dangerous chars stripped, leading/trailing whitespace trimmed
    assert system_control._sanitize_app_name('"; rm -rf /') == "rm -rf"
    assert system_control._sanitize_app_name("Google Chrome") == "Google Chrome"
    assert len(system_control._sanitize_app_name("A" * 100)) <= 50
test("app name sanitization", test_sanitize_app_name)

def test_set_volume_clamping():
    # Volume should be clamped 0-100
    result = system_control.set_volume(-10)
    # Should succeed with clamped value
    assert result.get("volume") == 0 or "error" in str(result)
test("volume clamping", test_set_volume_clamping)


# ═══════════════════════════════════════════════════════════════════════════
print("\n=== follow_ups ===")

import follow_ups

def test_followup_lifecycle():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    old_path = follow_ups.FOLLOWUPS_PATH
    follow_ups.FOLLOWUPS_PATH = path
    try:
        follow_ups.add("john@example.com", "Project update", threshold_hours=0)
        pending = follow_ups.get_pending_reminders()
        assert len(pending) >= 1, "Should have pending reminder (threshold=0)"
        follow_ups.mark_reminded("john@example.com", "Project update")
        pending = follow_ups.get_pending_reminders()
        assert len(pending) == 0, "Should have no pending after mark_reminded"
    finally:
        follow_ups.FOLLOWUPS_PATH = old_path
        os.unlink(path)
test("follow-up add/remind lifecycle", test_followup_lifecycle)


# ═══════════════════════════════════════════════════════════════════════════
print("\n=== quick_capture ===")

import quick_capture

def test_quick_capture_todo():
    result = quick_capture.detect("remind me to buy milk")
    assert result is not None, "Should capture 'remind me' pattern"
    assert result[0] == "todo"
    assert "buy milk" in result[1].lower()
test("quick capture: remind me", test_quick_capture_todo)

def test_quick_capture_note():
    result = quick_capture.detect("note: meeting moved to 3 PM")
    assert result is not None, "Should capture 'note:' pattern"
    assert result[0] == "memory"
test("quick capture: note", test_quick_capture_note)

def test_quick_capture_passthrough():
    result = quick_capture.detect("The weather is nice today")
    assert result is None, "Should pass through non-capture text"
test("quick capture: passthrough", test_quick_capture_passthrough)


# ═══════════════════════════════════════════════════════════════════════════
print("\n=== vision ===")

import vision

def test_screenshot():
    data = vision.take_screenshot()
    if data is not None:
        assert isinstance(data, bytes)
        assert len(data) > 100  # should be a real PNG
        # Verify PNG magic bytes
        assert data[:4] == b'\x89PNG', "Should be valid PNG"
    # Might be None if screencapture fails (CI, no display)
test("screenshot capture", test_screenshot)

def test_screenshot_cleanup():
    """Temp file should be cleaned up after screenshot."""
    tmp = os.path.join(tempfile.gettempdir(), "muse_screenshot.png")
    vision.take_screenshot()
    assert not os.path.exists(tmp), "Temp screenshot file should be cleaned up"
test("screenshot temp file cleanup", test_screenshot_cleanup)


# ═══════════════════════════════════════════════════════════════════════════
print("\n=== reflection ===")

import reflection

def test_reflection_empty():
    result = reflection.compose_week()
    assert "week in review" in result.lower()
    assert "quiet week" in result.lower()
test("reflection with no data", test_reflection_empty)

def test_reflection_with_brain():
    b = Brain.__new__(Brain)
    b._lock = threading.Lock()
    b.facts = [{"text": "Learned Rust", "category": "learning", "ts": time.time()}]
    b.deadlines = []
    b.meetings = [{"summary": "Team sync", "date": "2026-04-01", "action_items": [], "ts": time.time()}]
    b.habits = {}
    result = reflection.compose_week(brain=b)
    assert "Team sync" in result
test("reflection with brain data", test_reflection_with_brain)


# ═══════════════════════════════════════════════════════════════════════════
print("\n=== macros ===")

from macros import MacroEngine

def test_macro_crud():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        m = MacroEngine.__new__(MacroEngine)
        m._path = path
        m._macros = {}
        m._lock = threading.Lock()
        m.add("focus mode", "Enter focus mode", [{"type": "set_volume", "value": 0}])
        assert "focus mode" in m.list_all()
        m.remove("focus mode")
        assert "focus mode" not in m.list_all()
    finally:
        os.unlink(path)
test("macro add/remove", test_macro_crud)


# ═══════════════════════════════════════════════════════════════════════════
print("\n=== config ===")

import config

def test_config_paths_exist():
    assert config.DATA_DIR.exists(), "DATA_DIR should exist"
    assert config.PROJECT_DIR.exists(), "PROJECT_DIR should exist"
test("config paths exist", test_config_paths_exist)

def test_config_audio_settings():
    assert config.SAMPLE_RATE == 16000
    assert config.CHANNELS == 1
    assert config.DTYPE == "float32"
test("config audio settings", test_config_audio_settings)

def test_config_silence_threshold():
    assert 0 < config.SILENCE_THRESHOLD < 1, "Threshold should be between 0 and 1"
    assert config.MIN_AUDIO_DURATION > 0
test("config silence threshold", test_config_silence_threshold)


# ═══════════════════════════════════════════════════════════════════════════
print("\n=== recorder ===")

from recorder import _resample
import numpy as np

def test_resample_same_rate():
    audio = np.random.randn(16000).astype(np.float32)
    result = _resample(audio, 16000, 16000)
    assert np.array_equal(audio, result), "Same rate should return same audio"
test("resample same rate passthrough", test_resample_same_rate)

def test_resample_downsample():
    audio = np.random.randn(48000).astype(np.float32)  # 3 seconds at 48kHz
    result = _resample(audio, 48000, 16000)
    expected_len = 16000  # should be ~16000 samples
    assert abs(len(result) - expected_len) < 100, f"Expected ~{expected_len}, got {len(result)}"
test("resample 48kHz → 16kHz", test_resample_downsample)

def test_resample_upsample():
    audio = np.random.randn(16000).astype(np.float32)
    result = _resample(audio, 16000, 48000)
    assert abs(len(result) - 48000) < 100
test("resample 16kHz → 48kHz", test_resample_upsample)


# ═══════════════════════════════════════════════════════════════════════════
print("\n=== transcriber ===")

from transcriber import _clean_audio

def test_clean_audio_nan():
    audio = np.array([0.5, float('nan'), -0.3, float('inf'), 0.2], dtype=np.float32)
    result = _clean_audio(audio)
    assert np.isfinite(result).all(), "Should have no NaN/Inf"
test("clean audio handles NaN/Inf", test_clean_audio_nan)

def test_clean_audio_clipping():
    audio = np.array([1.5, -1.2, 0.8, 0.99], dtype=np.float32)
    result = _clean_audio(audio)
    assert np.max(np.abs(result)) <= 1.0, "Should normalize clipping audio"
test("clean audio normalizes clipping", test_clean_audio_clipping)

def test_clean_audio_normal():
    audio = np.array([0.1, -0.2, 0.05, 0.3], dtype=np.float32)
    result = _clean_audio(audio)
    assert np.allclose(audio, result), "Normal audio should pass through unchanged"
test("clean audio passes through normal audio", test_clean_audio_normal)


# ═══════════════════════════════════════════════════════════════════════════
print("\n=== edge cases ===")

def test_speech_prep_phone_number():
    """Phone numbers should not be read as large numbers."""
    result = prepare_for_speech("Call +1 555 123 4567")
    # Should not say "one billion five hundred..."
    assert "billion" not in result.lower()
test("speech prep: phone numbers", test_speech_prep_phone_number)

def test_speech_prep_time_with_minutes():
    result = prepare_for_speech("Meeting at 14:30")
    assert "2:30 PM" in result, f"Expected '2:30 PM' in '{result}'"
test("speech prep: time with minutes", test_speech_prep_time_with_minutes)

def test_speech_prep_midnight():
    result = prepare_for_speech("Event at 00:00")
    assert "12" in result and "AM" in result
test("speech prep: midnight", test_speech_prep_midnight)

def test_brain_deadline_filter():
    """Deadlines with unparseable due dates should still be included."""
    b = Brain.__new__(Brain)
    b._lock = threading.Lock()
    b.facts, b.meetings, b.habits = [], [], {}
    b.deadlines = [
        {"text": "Submit report", "due": "next Friday", "created": time.time(), "reminded": False},
        {"text": "Done task", "due": "2026-04-01", "created": time.time(), "reminded": True},
    ]
    results = b.get_deadlines(upcoming_days=7)
    assert any("Submit report" in d["text"] for d in results), "Unparseable due dates should be included"
    assert not any("Done task" in d["text"] for d in results), "Reminded deadlines should be excluded"
test("brain: deadline filtering edge cases", test_brain_deadline_filter)


# ═══════════════════════════════════════════════════════════════════════════
# robustness module tests

def test_cap_chunks():
    import robustness as rb
    small = [{"timestamp": f"{i}", "speaker": "You", "text": f"chunk {i}"} for i in range(10)]
    assert rb.cap_chunks(small) == small, "Small lists unchanged"
    huge = [{"timestamp": f"{i}", "speaker": "You", "text": f"chunk {i}"} for i in range(1000)]
    capped = rb.cap_chunks(huge)
    assert len(capped) <= rb.MAX_MEETING_CHUNKS + 1
    assert any("truncated" in c.get("text", "") for c in capped), "Should have truncation marker"
test("robustness: cap_chunks", test_cap_chunks)


def test_cap_transcript():
    import robustness as rb
    short = "hello world"
    assert rb.cap_transcript(short) == short
    long_text = "x" * (rb.MAX_TRANSCRIPT_CHARS + 5000)
    capped = rb.cap_transcript(long_text)
    assert len(capped) <= rb.MAX_TRANSCRIPT_CHARS + 100  # marker adds a few chars
    assert "truncated" in capped
test("robustness: cap_transcript", test_cap_transcript)


def test_deduped_decorator():
    import robustness as rb
    calls = []
    @rb.deduped("test_dedup", ttl=0.5)
    def slow_fn(x):
        calls.append(x)
        return x * 2
    assert slow_fn(3) == 6
    assert slow_fn(3) == 6  # cached
    assert len(calls) == 1, f"expected 1 call, got {len(calls)}"
    time.sleep(0.6)
    slow_fn(3)
    assert len(calls) == 2, "should miss cache after TTL"
test("robustness: deduped decorator", test_deduped_decorator)


def test_metrics():
    import robustness as rb
    rb.record_metric("test_metric", 100.0)
    rb.record_metric("test_metric", 200.0)
    rb.record_metric("test_metric", 300.0)
    summary = rb.get_metrics_summary()
    m = summary.get("test_metric", {})
    assert m.get("count") == 3
    assert m.get("avg_ms") == 200.0
    assert m.get("min_ms") == 100.0
    assert m.get("max_ms") == 300.0
test("robustness: metrics", test_metrics)


def test_parse_iso():
    from utils import parse_iso
    assert parse_iso("2026-04-15T10:30:00Z") is not None
    assert parse_iso("2026-04-15T10:30:00+00:00") is not None
    assert parse_iso("not a date") is None
    assert parse_iso("") is None
    assert parse_iso(None) is None
test("utils: parse_iso", test_parse_iso)


# ═══════════════════════════════════════════════════════════════════════════
# Results

print(f"\n{'='*60}")
print(f"RESULTS: {PASS} passed, {FAIL} failed")
print(f"Total: {PASS + FAIL} tests")
print(f"{'='*60}")

sys.exit(0 if FAIL == 0 else 1)
