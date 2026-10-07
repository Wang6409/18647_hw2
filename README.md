# Part 2: Tensor reordering

## 1. Build and test

On a Linux multicore machine with GCC and OpenMP:

```sh
g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -fopenmp -march=native -Isrc \
    src/reorder.cpp tests/reorder_test.cpp -o reorder_test
./reorder_test

g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -fopenmp -march=native -Isrc \
    src/reorder.cpp src/benchmark.cpp -o reorder_bench
```

The code implements index recalculation, iterative, and recursive CPU
reordering. The index version is also parallelized with OpenMP. Tests compare
the implementations against a reference.

## 2. Run on the selected multicore machine

Use an allocated ECE cluster node, AWS multi-vCPU instance, or PSC Bridges-2
machine. Record the hostname, CPU, available/allocated cores, RAM, and compiler
version. The 4 GiB input requires two 4 GiB buffers, so allow at least 8 GiB of
free memory. Replace `P` below with the cores allocated to your job.

Optional small test:

```sh
mkdir -p results
./reorder_bench --elements 1048576 --max-rank 6 --threads 4 \
    --repetitions 2 --warmups 0 --output results/smoke.csv
```

Required 4 GiB experiment (runs each rank with 1 through `P` threads):

```sh
./reorder_bench --elements 4294967296 --max-rank 32 --threads P \
    --repetitions 3 --warmups 1 --output results/reorder_4gib.csv
```

For `2^32` elements, the benchmark runs ranks 1 through 32. It distributes
the 32 factors of 2 as evenly as possible across each rank's dimensions; for
example, rank 3 uses `2048 x 2048 x 1024`. Dimensions therefore differ by at
most a factor of 2 and always multiply to `2^32`. It measures runtime with
`omp_get_wtime()` and also measures a parallel copy baseline.

## 3. Analyze and plot

```sh
python3 -m pip install matplotlib
python3 scripts/plot_results.py --input results/reorder_4gib.csv \
    --output-dir results/summary
```

This creates mean bandwidth and speedup CSVs, a memory-traffic model CSV,
`bandwidth_vs_cores.png` and `bandwidth_vs_cores.pdf` (one line per rank), and
`roofline.png` and `roofline.pdf` comparing reorder bandwidth with the measured
copy baseline. Submit `bandwidth_vs_cores.pdf` as the plot deliverable. If the
course Excel template is required, import `results/summary/bandwidth_vs_cores.csv`
into it, make its line chart use cores on the X axis, GB/s on the Y axis, and
one series per rank, then export the chart or workbook as PDF.

## 4. Report and submit

- Logical traffic: `2N` bytes (read `N`, write `N`); for `N = 2^32`, this is
  8 GiB. Effective bandwidth is `2N / time / 10^9` GB/s.
- Speedup is the one-thread mean time divided by the `P`-thread mean time.
- Rank affects destination locality: when traversing the innermost C-order
  dimension, the destination stride is
  `dims[0] * ... * dims[rank-2]` bytes. Physical traffic can exceed `2N`
  because of cache-line reads/writes; the traffic CSV reports an ideal
  write-allocate estimate of `3N`.
- Discuss scaling and bandwidth saturation using `roofline.png`; its copy
  baseline is an empirical ceiling, not a vendor peak specification.

Submit the raw benchmark CSV, generated summaries and plots, completed Excel
template, and machine/compiler details. Record actual measurements from the
selected machine; do not substitute the small test run for the 4 GiB results.
