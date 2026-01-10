# Profiling

The profiler tracks CPU/GPU operations and timing during training. It's enabled by default but limited to avoid memory issues.

## Quick Start

**Run training (profiling auto-enabled):**
```bash
uv run src/postings_classifier/train.py
```

**View results in TensorBoard:**
```bash
uv run tensorboard --logdir=profiling/traces --port=6006 --bind_all
# Open: http://localhost:6006/#pytorch_profiler
```

**View in Chrome trace viewer:**
- Open `chrome://tracing` in Chromium
- Load any `.pt.trace.json` from `profiling/traces/`

## Configuration

Settings in [configs/trainer/default.yaml](../../configs/trainer/default.yaml):
- `profiler: "pytorch"` — enable/disable ("null" to turn off)
- `profiler_record_memory: false` — record memory usage
- `profiler_wait: 0` — skip first N steps
- `profiler_warmup: 5` — warmup steps
- `profiler_active: 50` — record N steps
- `profiler_repeat: 1` — number of windows

## Common Usage

**Profile whole run (careful—may OOM):**
```bash
uv run src/postings_classifier/train.py \
  trainer.profiler_wait=0 \
  trainer.profiler_warmup=0 \
  trainer.profiler_active=100000
```

**Disable profiling:**
```bash
uv run src/postings_classifier/train.py trainer.profiler=null
```

**Advanced profiler (Python-level stats):**
```bash
uv run src/postings_classifier/train.py trainer.profiler=advanced
# View: cat profiling/run.txt
```

## Troubleshooting

**Out-of-memory (exit code 137)?** Reduce batch size or profiling window:
```bash
uv run src/postings_classifier/train.py data.batch_size=8 trainer.profiler_active=20
```

**No data in TensorBoard?** Check traces exist:
```bash
ls profiling/traces/
```
If empty, run training again.
