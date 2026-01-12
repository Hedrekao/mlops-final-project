# Running tests and coverage

Use the project's `uv` wrapper to run tests and print a coverage report.

- Run the test suite for the `tests/` folder:

```powershell
uv run pytest tests/
```

- Generate a coverage report (requires `coverage` to be installed in the environment):

```powershell
uv run coverage report -m
```

On Windows PowerShell use the above commands verbatim; in cmd.exe replace `uv run` with the appropriate wrapper if configured, or run `python -m pytest tests/` and `python -m coverage report -m` instead.
