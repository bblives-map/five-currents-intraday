from pathlib import Path
from setuptools import setup

setup(py_modules=sorted(path.stem for path in Path(__file__).parent.glob("*.py")
                        if not path.name.startswith("test_") and path.name not in {"setup.py", "conftest.py"}))
