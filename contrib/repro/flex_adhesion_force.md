# Adhesive flex: unresolved force parity

Run from the repository root after installing the development dependencies:

```sh
uv run --no-sync python contrib/repro/flex_adhesion_force.py --device cpu --adhesion 0
uv run --no-sync python contrib/repro/flex_adhesion_force.py --device cpu
uv run --no-sync python contrib/repro/flex_adhesion_force.py --device cpu --gap
```

Use `--device cuda:0` for GPU. The zero-adhesion control passes. The default run intentionally fails a normal-tolerance native-parity assertion: contact material initialization does not apply the missing flex-side generalized adhesion force. Reported contact force can agree while acceleration does not. `--gap` separately exposes missing separated contacts when using an adhesion gap rather than a matching contact margin.

This is a manual reproducer, not a passing regression or an expected-failure pytest test. The producer-only draft does not fix either physical mismatch. It does not attempt element, interpolated-flex, or passive/DISCRETE force semantics.
