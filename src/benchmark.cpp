#include "reorder.hpp"

#include <omp.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::size_t elements = std::size_t{1} << 32;
  int max_rank = 0;
  int max_threads = 0;
  int repetitions = 3;
  int warmups = 1;
  std::string output = "benchmark.csv";
};

std::size_t parse_size(const std::string& text, const std::string& name) {
  std::size_t parsed = 0;
  const unsigned long long value = std::stoull(text, &parsed);
  if (parsed != text.size() || value == 0 ||
      value > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument("invalid value for " + name + ": " + text);
  }
  return static_cast<std::size_t>(value);
}

int parse_positive_int(const std::string& text, const std::string& name) {
  const std::size_t value = parse_size(text, name);
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("value too large for " + name);
  }
  return static_cast<int>(value);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help") {
      std::cout
          << "Usage: reorder_bench [--elements POWER_OF_TWO] [--max-rank N] "
             "[--threads P] [--repetitions N] [--warmups N] [--output CSV]\n"
          << "Defaults: 2^32 bytes, all balanced ranks up to 32, all available "
             "OpenMP processors, 3 repetitions, 1 warmup.\n";
      std::exit(EXIT_SUCCESS);
    }
    if (i + 1 >= argc) {
      throw std::invalid_argument("missing value after " + arg);
    }
    const std::string value = argv[++i];
    if (arg == "--elements") {
      options.elements = parse_size(value, arg);
    } else if (arg == "--max-rank") {
      options.max_rank = parse_positive_int(value, arg);
    } else if (arg == "--threads") {
      options.max_threads = parse_positive_int(value, arg);
    } else if (arg == "--repetitions") {
      options.repetitions = parse_positive_int(value, arg);
    } else if (arg == "--warmups") {
      std::size_t parsed = 0;
      const unsigned long long warmups = std::stoull(value, &parsed);
      if (parsed != value.size()) {
        throw std::invalid_argument("invalid value for " + arg + ": " + value);
      }
      if (warmups > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("value too large for " + arg);
      }
      options.warmups = static_cast<int>(warmups);
    } else if (arg == "--output") {
      options.output = value;
    } else {
      throw std::invalid_argument("unknown option: " + arg);
    }
  }

  if ((options.elements & (options.elements - 1)) != 0) {
    throw std::invalid_argument("--elements must be a power of two");
  }
  int power = 0;
  for (std::size_t n = options.elements; n > 1; n >>= 1) {
    ++power;
  }
  if (options.max_rank == 0) {
    options.max_rank = power == 0 ? 1 : power;
  }
  if (options.max_rank > (power == 0 ? 1 : power)) {
    throw std::invalid_argument(
        "--max-rank cannot exceed log2(--elements)");
  }
  if (options.max_threads == 0) {
    options.max_threads = omp_get_num_procs();
  }
  if (options.max_threads < 1) {
    throw std::runtime_error("OpenMP reported no available processors");
  }
  if (options.output.empty()) {
    throw std::invalid_argument("--output must not be empty");
  }
  return options;
}

int log2_elements(std::size_t elements) {
  int power = 0;
  while (elements > 1) {
    elements >>= 1;
    ++power;
  }
  return power;
}

reorder::Dimensions square_dimensions(std::size_t elements, int rank) {
  const int power = log2_elements(elements);
  if (rank < 1 || rank > (power == 0 ? 1 : power)) {
    throw std::invalid_argument("rank is outside the supported range");
  }
  const int quotient = power / rank;
  const int remainder = power % rank;
  reorder::Dimensions dims(static_cast<std::size_t>(rank));
  for (int d = 0; d < rank; ++d) {
    const int exponent = quotient + (d < remainder ? 1 : 0);
    dims[static_cast<std::size_t>(d)] = std::size_t{1} << exponent;
  }
  return dims;
}

std::string dimensions_string(const reorder::Dimensions& dims) {
  std::string result;
  for (std::size_t d = 0; d < dims.size(); ++d) {
    if (d != 0) {
      result += 'x';
    }
    result += std::to_string(dims[d]);
  }
  return result;
}

double time_copy(const unsigned char* src, unsigned char* dst,
                 std::size_t elements, int threads) {
  const auto count = static_cast<std::int64_t>(elements);
  omp_set_num_threads(threads);
  const double start = omp_get_wtime();
#pragma omp parallel for schedule(static)
  for (std::int64_t i = 0; i < count; ++i) {
    dst[i] = src[i];
  }
  return omp_get_wtime() - start;
}

double time_reorder(const unsigned char* src, unsigned char* dst,
                    const reorder::Dimensions& dims, int threads) {
  omp_set_num_threads(threads);
  const double start = omp_get_wtime();
  reorder::by_index_parallel(src, dst, dims);
  return omp_get_wtime() - start;
}

double average(const std::vector<double>& values) {
  double total = 0.0;
  for (const double value : values) {
    total += value;
  }
  return total / static_cast<double>(values.size());
}

std::size_t innermost_destination_stride(
    const reorder::Dimensions& dims) {
  std::size_t stride = 1;
  for (std::size_t d = 0; d + 1 < dims.size(); ++d) {
    stride *= dims[d];
  }
  return stride;
}

void run(const Options& options) {
  if (options.elements >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument("--elements exceeds the OpenMP loop range");
  }
  std::cerr << "Allocating " << options.elements
            << " bytes each for source and destination ("
            << (2.0 * static_cast<double>(options.elements) /
                (1024.0 * 1024.0 * 1024.0))
            << " GiB total).\n";
  std::vector<unsigned char> src(options.elements);
  std::vector<unsigned char> dst(options.elements);

  const auto count = static_cast<std::int64_t>(options.elements);
#pragma omp parallel for schedule(static)
  for (std::int64_t i = 0; i < count; ++i) {
    src[i] = static_cast<unsigned char>(
        (static_cast<std::uint64_t>(i) * 131u + 17u) & 0xffu);
    dst[i] = 0;
  }

  std::vector<double> copy_seconds(
      static_cast<std::size_t>(options.max_threads + 1), 0.0);
  for (int threads = 1; threads <= options.max_threads; ++threads) {
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(options.repetitions));
    for (int repetition = 0; repetition < options.repetitions; ++repetition) {
      times.push_back(
          time_copy(src.data(), dst.data(), options.elements, threads));
    }
    copy_seconds[static_cast<std::size_t>(threads)] = average(times);
  }

  std::ofstream csv(options.output);
  if (!csv) {
    throw std::runtime_error("cannot open output CSV: " + options.output);
  }
  csv << "rank,dimensions,elements,threads,repetition,seconds,logical_bytes,"
         "write_allocate_ideal_bytes,bandwidth_GB_s,speedup,serial_mean_s,"
         "copy_mean_s,copy_bandwidth_GB_s,destination_inner_stride_bytes\n";
  csv << std::setprecision(10);

  const double logical_bytes = 2.0 * static_cast<double>(options.elements);
  for (int rank = 1; rank <= options.max_rank; ++rank) {
    const reorder::Dimensions dims = square_dimensions(options.elements, rank);
    std::cerr << "Benchmarking rank " << rank << " ("
              << dimensions_string(dims) << ")...\n";

    std::vector<std::vector<double>> times(
        static_cast<std::size_t>(options.max_threads + 1));
    for (int threads = 1; threads <= options.max_threads; ++threads) {
      omp_set_num_threads(threads);
      for (int warmup = 0; warmup < options.warmups; ++warmup) {
        (void)time_reorder(src.data(), dst.data(), dims, threads);
      }
      auto& samples = times[static_cast<std::size_t>(threads)];
      samples.reserve(static_cast<std::size_t>(options.repetitions));
      for (int repetition = 0; repetition < options.repetitions; ++repetition) {
        samples.push_back(time_reorder(src.data(), dst.data(), dims, threads));
      }
    }

    const double serial_mean = average(times[1]);
    const std::size_t destination_stride =
        innermost_destination_stride(dims);
    for (int threads = 1; threads <= options.max_threads; ++threads) {
      const double copy_mean =
          copy_seconds[static_cast<std::size_t>(threads)];
      const double copy_bandwidth = logical_bytes / copy_mean / 1.0e9;
      const auto& samples = times[static_cast<std::size_t>(threads)];
      for (std::size_t repetition = 0; repetition < samples.size();
           ++repetition) {
        const double seconds = samples[repetition];
        csv << rank << ',' << dimensions_string(dims) << ','
            << options.elements << ',' << threads << ',' << repetition + 1
            << ',' << seconds << ',' << logical_bytes << ','
            << 3.0 * static_cast<double>(options.elements) << ','
            << logical_bytes / seconds / 1.0e9 << ','
            << serial_mean / seconds << ',' << serial_mean << ',' << copy_mean
            << ',' << copy_bandwidth << ',' << destination_stride << '\n';
      }
    }
    csv.flush();
  }
  std::cerr << "Wrote measurements to " << options.output << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    omp_set_dynamic(0);
    const Options options = parse_options(argc, argv);
    run(options);
    return EXIT_SUCCESS;
  } catch (const std::bad_alloc&) {
    std::cerr << "Error: allocation failed; the benchmark needs two buffers "
                 "of the requested size.\n";
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
  }
  return EXIT_FAILURE;
}
