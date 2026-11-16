# Synthetic example data

These four Excel workbooks contain analytic signals with seeded random noise.
They contain **no participant measurements** and are not a validated model of
sweating, physiology, or an intervention. Their values and units are illustrative.

- Generator: `scripts/generate_synthetic_data.py`
- Default random seed: `2026`
- Sampling interval: 10 seconds
- Record duration: 55 minutes
- Channels: `Na`, `K`, `Ca`, `TEMP`, `FR`, and `HR`
- Provenance marker: `SYNTHETIC_DATA.json`

Run from the repository root:

```bash
python scripts/run_all.py --config configs/synthetic_smoke.yaml
```

To replace these generated examples with the default seeded signals:

```bash
python scripts/generate_synthetic_data.py --force
```

The seed reproduces numeric signal values, not necessarily identical Excel ZIP
bytes or workbook timestamps. Results are software checks, not paper metrics.
