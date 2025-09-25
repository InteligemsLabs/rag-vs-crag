#!/usr/bin/env python3
"""
Simple script to run the RAG vs CRAG comparison app.
"""
import subprocess
import sys

def main():
    """Run the Streamlit app."""
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
