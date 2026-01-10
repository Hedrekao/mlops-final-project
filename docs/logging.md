# Logging

This project uses [Loguru](https://github.com/Delgan/loguru) for application logging.

## How it works
- Logging is initialized inside the training script via `setup_logging` in `src/postings_classifier/logging_utils.py`.
- Output goes to stdout and a rotating file at `logs/app.log` (created automatically).
- Default level is `INFO`.

## Run with logging
```bash
uv run src/postings_classifier/train.py
```
Logs appear in the terminal and in `logs/app.log`.

## Change logging behavior
- Level: change the `level` argument in `setup_logging` (e.g., `"DEBUG"`, `"WARNING"`).
- Rotation: change the `rotation` argument (e.g., `"10 MB"`, `"1 day"`).
- Disable file logging: remove the file handler in `setup_logging`.

## Quick reference
- Logging setup: `src/postings_classifier/logging_utils.py`
- Log file: `logs/app.log`
- Default level: `INFO`
- Rotates at: `10 MB`
