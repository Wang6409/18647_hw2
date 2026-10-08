#include "reorder.hpp"

#include <omp.h>

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::size_t element_count(const reorder::Dimensions& dims) {
  std::size_t count = 1;
  for (const std::size_t dim : dims) {
    count *= dim;
  }
  return count;
}

std::vector<unsigned char> reference(
    const std::vector<unsigned char>& src, const reorder::Dimensions& dims) {
  const std::size_t count = element_count(dims);
  std::vector<std::size_t> c_strides(dims.size());
  std::vector<std::size_t> f_strides(dims.size());

  std::size_t stride = 1;
  for (std::size_t d = dims.size(); d-- > 0;) {
    c_strides[d] = stride;
    stride *= dims[d];
  }
  stride = 1;
  for (std::size_t d = 0; d < dims.size(); ++d) {
    f_strides[d] = stride;
    stride *= dims[d];
  }

  std::vector<unsigned char> expected(count);
  for (std::size_t f_offset = 0; f_offset < count; ++f_offset) {
    std::size_t remaining = f_offset;
    std::size_t c_offset = 0;
    for (std::size_t d = dims.size(); d-- > 0;) {
      const std::size_t index = remaining / f_strides[d];
      remaining %= f_strides[d];
      c_offset += index * c_strides[d];
    }
    expected[f_offset] = src[c_offset];
  }
  return expected;
}

void check_case(const reorder::Dimensions& dims) {
  const std::size_t count = element_count(dims);
  std::vector<unsigned char> src(count);
  for (std::size_t i = 0; i < count; ++i) {
    src[i] = static_cast<unsigned char>((i * 37 + 11) % 256);
  }

  const std::vector<unsigned char> expected = reference(src, dims);
  std::vector<unsigned char> output(count);
  const std::vector<void (*)(const unsigned char*, unsigned char*,
                             const reorder::Dimensions&)>
      algorithms = {reorder::by_index,
                    reorder::by_index_parallel,
                    reorder::by_iteration,
                    reorder::by_recursion};
  const std::vector<std::string> names = {"index", "parallel index",
                                          "iteration", "recursion"};

  for (std::size_t i = 0; i < algorithms.size(); ++i) {
    algorithms[i](src.data(), output.data(), dims);
    if (output != expected) {
      throw std::runtime_error("algorithm " + names[i] +
                               " failed for rank " +
                               std::to_string(dims.size()));
    }
  }
}

void check_invalid_inputs() {
  unsigned char src[1] = {0};
  unsigned char dst[1] = {0};
  bool rejected = false;
  try {
    reorder::by_index(src, dst, {});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) {
    throw std::runtime_error("empty dimensions were not rejected");
  }

  rejected = false;
  try {
    reorder::by_iteration(src, dst, {2, 0});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) {
    throw std::runtime_error("zero-sized dimensions were not rejected");
  }

  rejected = false;
  try {
    reorder::by_recursion(src, dst, {std::numeric_limits<std::size_t>::max(),
                                    2});
  } catch (const std::overflow_error&) {
    rejected = true;
  }
  if (!rejected) {
    throw std::runtime_error("overflowing element counts were not rejected");
  }

  rejected = false;
  try {
    reorder::by_recursion(src, src, {1});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) {
    throw std::runtime_error("identical source and destination were accepted");
  }
}

}  // namespace

int main() {
  try {
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    const std::vector<reorder::Dimensions> cases = {
        {1},          {7},          {2, 3},        {3, 1, 4},
        {2, 3, 4},    {2, 2, 3, 2}, {2, 3, 2, 2, 2},
        {2, 2, 2, 2, 2, 2}, {257, 257}};
    for (const auto& dims : cases) {
      check_case(dims);
    }
    check_invalid_inputs();
    std::cout << "All reorder tests passed (" << cases.size()
              << " dimension sets, four implementations).\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "Test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
