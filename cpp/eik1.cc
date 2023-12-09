// References
// [1]
// https://www.ams.org/journals/mcom/2005-74-250/S0025-5718-04-01678-3/viewer/#ltxid5
#include <format>
#include <iostream>
#include <span>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "util.hh"

namespace nb = nanobind;

using std::span;

template <typename T>
using ndarray2 = nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

float compute_u_cand(float a, float b, const float s) {
  // eq. 2.4 from [1]
  if (std::abs(a - b) >= s) {
    return std::min(a, b) + s;
  } else {
    return 0.5f * (a + b + std::sqrt(2.0f * s * s - (a - b) * (a - b)));
  }
}

void sweep_lr(span<float> uu, span<const float> uu0, span<const float> uu1,
              span<const float> ss) {
  CHECK(uu.size() == uu0.size());
  CHECK(uu.size() == uu1.size());
  CHECK(uu.size() == ss.size());

  const auto width = uu.size();

  for (int i = 1; i < width - 1; ++i) {
    // min value from above/below
    const auto b = std::min(uu0[i], uu1[i]);

    // min value from left/right
    const auto a = std::min(uu[i - 1], uu[i + 1]);

    // compute candidate value (i.e. min value coming from the current sweep
    // direction)
    const auto u_cand = compute_u_cand(a, b, ss[i]);

    uu[i] = std::min(uu[i], u_cand);
  }
}

void sweep_rl(span<float> uu, span<const float> uu0, span<const float> uu1,
              span<const float> ss) {
  CHECK(uu.size() == uu0.size());
  CHECK(uu.size() == uu1.size());
  CHECK(uu.size() == ss.size());

  const auto width = uu.size();

  for (int i = width - 2; i > 0; --i) {
    // min value from above/below
    const auto b = std::min(uu0[i], uu1[i]);

    // min value from left/right
    const auto a = std::min(uu[i - 1], uu[i + 1]);

    // compute candidate value (i.e. min value coming from the current sweep
    // direction)
    const auto u_cand = compute_u_cand(a, b, ss[i]);

    uu[i] = std::min(uu[i], u_cand);
  }
}

// template<typename T>
// using ndarray2_view =
// std::remove_reference_t<decltype(ndarray2<T>().view())>;

// template<typename T>
// span<T> row(ndarray2_view<T> arr, int i) {
//   return span{arr.data() + i * arr.shape(1), arr.shape(1)};
// }

template <typename T> auto row(ndarray2<T> arr, int i) {
  return span{arr.data() + i * arr.shape(1), arr.shape(1)};
}

void solve(ndarray2<const float> slowness, ndarray2<float> tt, int num_iters) {
  if (slowness.shape(1) != slowness.stride(0)) {
    // NOTE EDF the row() function assumes this...obviously it would be easy not
    // to require this by using the stride, but it's not necessary at the moment
    throw std::runtime_error("array must be packed");
  }

  // check that the arrays have the same shape
  if (slowness.shape(0) != tt.shape(0) || slowness.shape(1) != tt.shape(1)) {
    throw std::runtime_error("shape mismatch");
  }

  const auto nj = slowness.shape(0);

  for (int iter = 0; iter < num_iters; ++iter) {
    for (int j = 1; j < nj - 1; ++j) {
      sweep_lr(row(tt, j), row(tt, j - 1), row(tt, j + 1), row(slowness, j));
    }

    for (int j = 1; j < nj - 1; ++j) {
      sweep_rl(row(tt, j), row(tt, j - 1), row(tt, j + 1), row(slowness, j));
    }

    for (int j = nj - 2; j > 0; --j) {
      sweep_lr(row(tt, j), row(tt, j - 1), row(tt, j + 1), row(slowness, j));
    }

    for (int j = nj - 2; j > 0; --j) {
      sweep_rl(row(tt, j), row(tt, j - 1), row(tt, j + 1), row(slowness, j));
    }
  }
}

NB_MODULE(eik1, m) { m.def("solve", &solve); }