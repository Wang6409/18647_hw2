#ifndef REORDER_HPP
#define REORDER_HPP

#include <cstddef>
#include <vector>

namespace reorder {

using Dimensions = std::vector<std::size_t>;

void by_index(const unsigned char* src, unsigned char* dst,
              const Dimensions& dims);
void by_index_parallel(const unsigned char* src, unsigned char* dst,
                       const Dimensions& dims);
void by_iteration(const unsigned char* src, unsigned char* dst,
                  const Dimensions& dims);
void by_recursion(const unsigned char* src, unsigned char* dst,
                  const Dimensions& dims);

}  // namespace reorder

#endif
