# FINC450 Lifecycle Investment Strategy

## Requirements

- Python 3.8+

## Installation

Install all required packages:

```bash
pip install -r requirements.txt
```

## Generating PDFs

To regenerate the lifecycle strategy PDF:

```bash
python3 lifecycle_strategy.py
```

This creates `lifecycle_strategy.pdf` with the full lifecycle investment analysis.

### Custom Parameters

```bash
python3 lifecycle_strategy.py -o custom.pdf --initial-earnings 120 --stock-beta 0.4 --bond-duration 5.0
```

Run `python3 lifecycle_strategy.py --help` for all available options.
