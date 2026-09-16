"""CardioRAG Streamlit demo — bridge to the canonical v2.4 3-tab app.

Run either:
    streamlit run app/streamlit_app.py   (canonical)
    streamlit run streamlit_app.py       (this bridge, same app)
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))

import streamlit_app as _canonical  # noqa: F401, E402 — executes the canonical app
