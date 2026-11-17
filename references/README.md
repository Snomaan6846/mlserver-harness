# References

Frozen parity outputs — the correctness contract. Every trial compares
its responses byte-for-byte against these; any mismatch fails the
trial. Captured once, deliberately, via
`infrastructure/capture-references.sh` (never during a trial).

Missing references for a model are captured automatically on first
`make setup`. See `docs/METHODOLOGY.md`.
