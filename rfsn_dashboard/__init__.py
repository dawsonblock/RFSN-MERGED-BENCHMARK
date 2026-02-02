"""RFSN Dashboard package.

Install with: pip install rfsn-controller[dashboard]
Run with: streamlit run rfsn_dashboard/app.py
"""

__all__ = ["main"]


def main() -> None:
    """Entry point for the dashboard.
    
    Import streamlit lazily to avoid requiring it just to use other parts
    of the rfsn package.
    """
    try:
        import streamlit.web.cli as stcli
        import sys
        from pathlib import Path
        
        app_path = Path(__file__).parent / "app.py"
        sys.argv = ["streamlit", "run", str(app_path)]
        stcli.main()
    except ImportError as e:
        print(f"Error: streamlit is not installed. Install with: pip install rfsn-controller[dashboard]")
        print(f"Details: {e}")
        raise SystemExit(1)
