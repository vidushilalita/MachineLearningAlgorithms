"""
Entrypoint – run with: streamlit run app.py
"""

import subprocess, sys
from pathlib import Path

if __name__ == "__main__":
    app = Path(__file__).parent / "src" / "streamlit_app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)])
