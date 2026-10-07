#include "reorder.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace reorder {
namespace {

struct Layout {
  std::size_t element_count;
  Dimensions c_strides;
  Dimensions f_strides;
};

Layout make_layout(const Dimensions& dims) {
  if (dims.empty()) {
    throw std::invalid_argument("rank must be at least 1");
  }

  Layout layout{1, Dimensions(dims.size()), Dimensions(dims.size())};
  for (const std::size_t dim : dims) {
    if (dim == 0) {
      throw std::invalid_argument("dimensions must be positive");
    }
    if (layout.element_count >
        std::numeric_limits<std::size_t>::max() / dim) {
      throw std::overflow_error("element count overflows size_t");
    }
    layout.element_count *= dim;
  }

  std::size_t stride = 1;
  for (std::size_t d = dims.size(); d-- > 0;) {
    layout.c_strides[d] = stride;
    stride *= dims[d];
  }

  stride = 1;
  for (std::size_t d = 0; d < dims.size(); ++d) {
    layout.f_strides[d] = stride;
    stride *= dims[d];
  }

  return layout;
}

void validate_buffers(const unsigned char* src, unsigned char* dst) {
  if (src == nullptr || dst == nullptr) {
    throw std::invalid_argument("source and destination must not be null");
  }
  if (src == dst) {
    throw std::invalid_argument(
        "source and destination must not be the same pointer");
  }
}

void recurse(const unsigned char* src, unsigned char* dst,
             const Dimensions& dims, const Layout& layout, std::size_t dim,
             std::size_t c_offset, std::size_t f_offset) {
  if (dim + 1 == dims.size()) {
    for (std::size_t i = 0; i < dims[dim]; ++i) {
      dst[f_offset + i * layout.f_strides[dim]] = src[c_offset + i];
    }
    return;
  }

  for (std::size_t i = 0; i < dims[dim]; ++i) {
    recurse(src, dst, dims, layout, dim + 1,
            c_offset + i * layout.c_strides[dim],
            f_offset + i * layout.f_strides[dim]);
  }
}

}  // namespace

void by_index(const unsigned char* src, unsigned char* dst,
              const Dimensions& dims) {
  const Layout layout = make_layout(dims);
  validate_buffers(src, dst);

  for (std::size_t c_offset = 0; c_offset < layout.element_count; ++c_offset) {
    std::size_t f_offset = 0;
    for (std::size_t d = 0; d < dims.size(); ++d) {
      const std::size_t index =
          (c_offset / layout.c_strides[d]) % dims[d];
      f_offset += index * layout.f_strides[d];
    }
    dst[f_offset] = src[c_offset];
  }
}

void by_index_parallel(const unsigned char* src, unsigned char* dst,
                       const Dimensions& dims) {
  const Layout layout = make_layout(dims);
  validate_buffers(src, dst);
  if (layout.element_count >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("element count exceeds parallel loop range");
  }

  const auto count = static_cast<std::int64_t>(layout.element_count);
#pragma omp parallel for schedule(static)
  for (std::int64_t c_offset = 0; c_offset < count; ++c_offset) {
    std::size_t f_offset = 0;
    for (std::size_t d = 0; d < dims.size(); ++d) {
      const std::size_t index =
          (static_cast<std::size_t>(c_offset) / layout.c_strides[d]) %
          dims[d];
      f_offset += index * layout.f_strides[d];
    }
    dst[f_offset] = src[c_offset];
  }
}

void by_iteration(const unsigned char* src, unsigned char* dst,
                  const Dimensions& dims) {
  const Layout layout = make_layout(dims);
  validate_buffers(src, dst);

  Dimensions indices(dims.size(), 0);
  std::size_t c_offset = 0;
  std::size_t f_offset = 0;

  for (std::size_t copied = 0; copied < layout.element_count; ++copied) {
    dst[f_offset] = src[c_offset];
    if (copied + 1 == layout.element_count) {
      break;
    }

    for (std::size_t d = dims.size(); d-- > 0;) {
      if (indices[d] + 1 < dims[d]) {
        ++indices[d];
        c_offset += layout.c_strides[d];
        f_offset += layout.f_strides[d];
        break;
      }
      indices[d] = 0;
      c_offset -= (dims[d] - 1) * layout.c_strides[d];
      f_offset -= (dims[d] - 1) * layout.f_strides[d];
    }
  }
}

void by_recursion(const unsigned char* src, unsigned char* dst,
                  const Dimensions& dims) {
  const Layout layout = make_layout(dims);
  validate_buffers(src, dst);
  recurse(src, dst, dims, layout, 0, 0, 0);
}

}  // namespace reorder
