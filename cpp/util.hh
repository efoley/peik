#pragma once

#include <stdexcept>

// need 2 macros so that any macro passed as argument will be expanded
// before being stringified
#define _STRINGIZE_DETAIL(x) #x
#define _STRINGIZE(x) _STRINGIZE_DETAIL(x)

#define CHECK(expr)                                                            \
  ygg::_check(expr, "(CHECK failed at " __FILE__                               \
                    ":" _STRINGIZE(__LINE__) "): " #expr)

#define DEBUG_CHECK(expr) CHECK(expr)

namespace ygg {

inline void _check(bool b, const char *msg) {
  if (!b) {
    throw std::runtime_error(msg);
  }
}

#define FAIL(msg)                                                              \
  ygg::_fail("(FAIL at " __FILE__ ":" _STRINGIZE(__LINE__) "): " msg)

[[noreturn]] inline void _fail(const char *msg) {
  throw std::runtime_error(msg);
}

#define UNREACHABLE()                                                          \
  ygg::_unreachable("(UNREACHABLE at " __FILE__ ":" _STRINGIZE(__LINE__) ")")

[[noreturn]] inline void _unreachable(const char *msg) {
  throw std::runtime_error(msg);
}

} // namespace ygg