"""Launch PaperPilot: ``streamlit run app.py``.

Streamlit runs the file it is given as a script, so this thin entry point keeps
the interface itself inside the package where it can be imported and tested.
"""

import runpy

runpy.run_module("paperpilot.ui.app", run_name="__main__")
