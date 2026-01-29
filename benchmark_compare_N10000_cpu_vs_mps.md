# N=10,000 Benchmark Comparison (CPU vs MPS)

Sources:
- CPU: `benchmark_results_N10000.csv`
- MPS (with CPU fallback for `poisson`): `benchmark_results_N10000_mps.csv`

Notes:
- MPS run uses `PYTORCH_ENABLE_MPS_FALLBACK=1` because `torch.poisson` is not supported on MPS.
- FastGEMF is CPU-only in both runs, so its MPS numbers reflect CPU execution.

## BarabasiAlbert

| Algorithm | CPU mean time (s) | MPS mean time (s) | MPS/CPU | CPU mean L1 | MPS mean L1 |
| --- | --- | --- | --- | --- | --- |
| Gillespie_v1 | 0.1729 | 0.8058 | 4.66x | 0.0 | 0.0 |
| SystemWiseTauLeaping_v1 | 0.1723 | 1.6187 | 9.39x | 1019.3 | 360.7 |
| HeapTauLeaping_v1 | 0.0516 | 0.6430 | 12.46x | 340.7 | 362.0 |
| FastGEMF | 0.8048 | 0.7877 | 0.98x | 6574.7 | 6592.7 |

## WattsStrogatz

| Algorithm | CPU mean time (s) | MPS mean time (s) | MPS/CPU | CPU mean L1 | MPS mean L1 |
| --- | --- | --- | --- | --- | --- |
| Gillespie_v1 | 0.2189 | 0.7780 | 3.56x | 0.0 | 0.0 |
| SystemWiseTauLeaping_v1 | 0.1396 | 1.5172 | 10.87x | 560.7 | 227.3 |
| HeapTauLeaping_v1 | 0.1118 | 0.4728 | 4.23x | 332.7 | 133.3 |
| FastGEMF | 0.5011 | 0.4717 | 0.94x | 3276.7 | 3244.7 |

## Takeaways

- MPS is slower here because `poisson` falls back to CPU, so timings are mixed CPU/MPS and not representative of a true GPU speedup.
- Heap tau-leaping remains fastest on CPU for N=10k; on MPS its advantage is reduced due to fallback overhead.
- L1 differences are in the same order of magnitude across CPU vs MPS runs, but variance is high with only 3 runs.

