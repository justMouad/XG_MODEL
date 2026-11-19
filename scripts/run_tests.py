"""Dependency-free test runner (use `pytest` normally; this exists for minimal environments)."""
import importlib, sys, traceback, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))
ok = bad = 0
for f in sorted(pathlib.Path(__file__).resolve().parents[1].glob("tests/test_*.py")):
    mod = importlib.import_module(f.stem)
    for name in dir(mod):
        if name.startswith("test_"):
            try:
                getattr(mod, name)(); ok += 1; print("PASS", f.stem, name)
            except Exception:
                bad += 1; print("FAIL", f.stem, name); traceback.print_exc()
print(f"\n{ok} passed, {bad} failed")
sys.exit(1 if bad else 0)
