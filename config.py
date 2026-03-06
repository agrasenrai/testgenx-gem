# config.py -- All tunable settings in one place
import os

# -- Gemini API (replacing all local ML models) --------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBdlymiZXP-KtW9yhwfG_Y3LL6MZASvISY")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# ── Paths ─────────────────────────────────────────────────────────────────────
LINGUISTIC_REGISTRY_PATH = "linguistic_registry"
OUTPUT_DIR = "output"

# ── BVA offset: how many steps from boundary to generate "well inside" values ──
BVA_WELL_INSIDE_OFFSET = 5

# ── Optional: GPT-4o edge case augmentation ───────────────────────────────────
OPENAI_MODEL   = "gpt-4o"
EDGE_CASES_PER_TC = 5
