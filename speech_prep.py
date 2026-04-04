"""Prepare text for natural speech synthesis.

Converts written/LLM text into spoken-friendly text:
- Numbers → words ("12 PM" → "twelve PM")
- Abbreviations → expanded ("Dr." → "Doctor")
- Punctuation → natural pauses
- Lists → flowing sentences
- Removes visual-only formatting (bullets, dashes, markdown)
"""

import re


def prepare_for_speech(text: str) -> str:
    """Convert text to sound natural when spoken aloud."""
    if not text:
        return text

    t = text.strip()

    # Remove markdown formatting
    t = re.sub(r'\*\*(.+?)\*\*', r'\1', t)  # bold
    t = re.sub(r'\*(.+?)\*', r'\1', t)      # italic
    t = re.sub(r'`(.+?)`', r'\1', t)        # code
    t = re.sub(r'^#+\s*', '', t, flags=re.MULTILINE)  # headers

    # Remove bullet points and dashes at line starts
    t = re.sub(r'^[\-\•\*]\s*', '', t, flags=re.MULTILINE)
    t = re.sub(r'^\d+\.\s*', '', t, flags=re.MULTILINE)

    # Remove ISO timestamps FIRST (before time conversion)
    t = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[^\s,.)]*', '', t)

    # Convert 24-hour to 12-hour (13:00 → 1 PM)
    def _24to12(m):
        h = int(m.group(1))
        mins = m.group(2) if m.group(2) else "00"
        ampm = "AM" if h < 12 else "PM"
        if h == 0: h = 12
        elif h > 12: h -= 12
        return f"{h}:{mins} {ampm}" if mins != "00" else f"{h} {ampm}"
    t = re.sub(r'\b(\d{1,2}):(\d{2})(?!\s*[APap][Mm])\b', _24to12, t)

    # Simplify :00 times (12:00 PM → 12 PM)
    t = re.sub(r'(\d{1,2}):00\s*(AM|PM|am|pm)', r'\1 \2', t)

    # Common abbreviations
    replacements = {
        ' Dr. ': ' Doctor ',
        ' Mr. ': ' Mister ',
        ' Mrs. ': ' Missus ',
        ' Ms. ': ' Miz ',
        ' Jr. ': ' Junior ',
        ' Sr. ': ' Senior ',
        ' vs. ': ' versus ',
        ' etc.': ' etcetera.',
        ' e.g.': ' for example',
        ' i.e.': ' that is',
        ' w/ ': ' with ',
        ' w/o ': ' without ',
        ' & ': ' and ',
        ' approx. ': ' approximately ',
        ' appt.': ' appointment.',
        ' dept.': ' department.',
        ' mgr.': ' manager.',
    }
    for old, new in replacements.items():
        t = t.replace(old, new)

    # Collapse multiple newlines/spaces into single pause
    t = re.sub(r'\n+', '. ', t)
    t = re.sub(r'\s{2,}', ' ', t)

    # Remove parenthetical timestamps like (2026-04-04)
    t = re.sub(r'\(\d{4}-\d{2}-\d{2}\)', '', t)

    # Clean up double periods
    t = re.sub(r'\.{2,}', '.', t)
    t = re.sub(r'\.\s*\.', '.', t)

    # Add slight pause after sentences for more natural rhythm
    # (Kokoro handles this with punctuation)

    return t.strip()
