# HW2 Part 2

Requirements: GCC with OpenMP and Python 3 with Matplotlib.

Machine: ECE cluster ece006 (Intel Xeon E5-2640 v4, 10 physical cores, 2 hardware threads per core, 128 GB RAM, GCC 8.5.0).

## Build and test

```sh
g++ -std=c++17 -O3 -fopenmp -march=native -Isrc \
    src/reorder.cpp tests/reorder_test.cpp -o reorder_test

./reorder_test

g++ -std=c++17 -O3 -fopenmp -march=native -Isrc \
    src/reorder.cpp src/benchmark.cpp -o reorder_bench
```

## Run the 4 GiB benchmark

Run on an allocated multicore machine with at least 8 GiB free memory. Replace
`P` with the number of allocated cores:

```sh
mkdir -p results
./reorder_bench --elements 4294967296 --max-rank 32 --threads P \
    --repetitions 1 --warmups 0 --output results/reorder_4gib.csv
```

For `2^32` elements, only exact-square ranks `1, 2, 4, 8, 16, 32` are tested,
with thread counts from 1 to `P`.

## Generate plots

```sh
python3 -m pip install matplotlib
python3 scripts/plot_results.py --input results/reorder_4gib.csv \
    --output-dir results/summary
```

Import `results/summary/bandwidth_vs_cores.csv` into the course Excel template
and save the required plot as `.xlsx` or `.pdf`.

## Files and experiments

- `src/`: the three reorder algorithms, OpenMP implementation, and benchmark.
- `tests/`, `scripts/`: correctness tests and CSV plotting script.
- `runs/`: ECE006 machine/test log, completed 4 GiB experiment (ranks 1, 2, 4,
  8, 16, 32; 1–20 threads; one measurement per setting), and the 1 MiB smoke
  run.
  ```sh
    { echo "=== Machine Info ==="; hostname; lscpu | grep -E "^CPU\(s\)|Model name|Thread|Core"; \
        free -h; gcc --version | head -1; echo ""; echo "=== Test Output ==="; ./reorder_test; \
        } > results/machine_and_test.log 2>&1

    time ./reorder_bench --elements 1048576 --max-rank 16 --threads 4 \
        --repetitions 1 --warmups 0 --output results/smoke_ece006.csv

    wc -l results/smoke_ece006.csv

    nohup ./reorder_bench --elements 4294967296 --max-rank 32 --threads 20 --repetitions 1 --warmups 0 \
        --output results/reorder_4gib_ece006.csv > results/reorder_4gib_ece006.log 2>&1 < /dev/null &

    python3 scripts/plot_results.py --input results/reorder_4gib_ece006.csv  \
        --output-dir results/summary_ece006
    
    scp -r fangmiaw@ece006.ece.local.cmu.edu:~/647_hw2/results ./runs
  ```
- `plots/bandwidth_vs_cores.csv`: effective bandwidth from the 4 GiB ECE006
  run, arranged by rank and thread count for the required plot.
- `plots/bandwidth_vs_cores.pdf`: bandwidth-versus-thread-count plot with one
  line per exact-square rank (1, 2, 4, 8, 16, 32); use these data in the course
  Excel template for the required final chart.
- `plots/speedup_vs_cores.csv`: speedup relative to one thread for each rank
  and thread count, supporting the parallel speedup analysis.
- `plots/traffic_model.csv`: logical and idealized write-allocate traffic,
  plus destination stride by rank, supporting the memory-traffic discussion.
- `plots/roofline.pdf`: supplementary comparison of reorder bandwidth with
  the measured contiguous-copy bandwidth baseline.
- `ai/`: gemini & chatGPT
