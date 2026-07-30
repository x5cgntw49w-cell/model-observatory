# Contributing

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

Keep changes focused, add tests for behavioral changes, and preserve the
dependency-light runtime. Generated reports must remain deterministic for a
fixed input and configuration.

## Data and metrics

- Never commit private or production prediction data.
- Document any new metric's definition and directionality.
- Add a hand-calculated test case before optimizing an implementation.
- Treat drift scores as diagnostic signals, not universal pass/fail rules.
