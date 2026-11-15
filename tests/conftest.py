import os
import tempfile

# the app's defaults file (File → 설정) of this computer must not affect the tests
os.environ["BLOT_PROFILER_PREFS"] = os.path.join(tempfile.mkdtemp(prefix="blotprefs_"), "prefs.json")
os.environ.setdefault("BLOT_PROFILER_LANG", "ko")  # the tests check the Korean text
