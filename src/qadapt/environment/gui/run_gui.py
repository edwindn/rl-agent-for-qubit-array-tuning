#!/usr/bin/env python3
"""
Run script for the qarray parameter exploration GUI.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the Streamlit GUI."""

    # Get the directory containing this script
    gui_dir = Path(__file__).parent

    # Change to the GUI directory so imports work correctly
    os.chdir(gui_dir)

    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "streamlit_app.py",
            "--server.port", "8502",
            "--server.address", "localhost"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nGUI stopped by user")
        return 0

    return 0

if __name__ == "__main__":
    exit(main())