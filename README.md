# function-calling-sft-3b

> **Era:** 2023–2024 · Senior AI Engineer focus area (agents). Technique from 2023, reimplemented in 2026.

Synthetic multi-turn typed-tool trajectories, strict schema validation, and trainable function-calling models. Tiny CPU proxy results and the unexecuted configurable 3B GPU workflow are documented separately as those stages land.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
PYTHONPATH=src python scripts/smoke.py
```

## License

MIT
