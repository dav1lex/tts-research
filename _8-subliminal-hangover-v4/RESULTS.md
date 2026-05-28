# Stage B Results — Tempo Acceleration & Pitch Compression

## Design

N=594: 3 models (Chatterbox, Kokoro, XTTS-v2) x 10 targets x 10 reps, plus control rows.
Mixed-effects models: `speaking_rate ~ condition` and `f0_cv ~ condition` with random intercepts.

## Primary Finding — Tempo Acceleration

`s peaking_rate ~ condition`, mixed effects:

| Model | Coef | 95% CI | p |
|-------|------|--------|---|
| Chatterbox | +0.352 | [+0.103, +0.601] | 0.006 |
| Kokoro | +1.356 | [+1.157, +1.555] | 1.5e-40 |
| XTTS | +0.704 | [+0.391, +1.016] | 1e-05 |

All CIs exclude zero. Effect confirmed across all three architectures.

## Secondary Finding — Pitch Compression

`f0_cv ~ condition`, mixed effects:

| Model | Coef | 95% CI | p | Verdict |
|-------|------|--------|---|---------|
| Kokoro | -0.045 | [-0.061, -0.029] | 2e-08 | Confirmed |
| Chatterbox | -0.006 | — | 0.668 | Null (convergence failures — unreliable estimate) |
| XTTS | -0.004 | — | 0.762 | Null |

## Conclusion

Robotic number primes cause consistent **tempo acceleration** across all tested TTS architectures.
Pitch compression is a **Kokoro-specific phenotype**.
