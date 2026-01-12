import os

_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data

# Expose the test package path constants for test modules importing them
__all__ = ["_TEST_ROOT", "_PROJECT_ROOT", "_PATH_DATA"]

# Also provide a public alias for convenience
PATH_DATA = _PATH_DATA
