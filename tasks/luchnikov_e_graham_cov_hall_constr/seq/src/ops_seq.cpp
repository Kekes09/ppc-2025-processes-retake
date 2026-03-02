#include "luchnikov_e_graham_cov_hall_constr/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stack>
#include <utility>
#include <vector>

#include "luchnikov_e_graham_cov_hall_constr/common/include/common.hpp"
#include "util/include/util.hpp"

namespace luchnikov_e_graham_cov_hall_constr {

namespace {
struct Point {
  double x;
  double y;
  int index;

  Point(double x_val, double y_val, int idx) : x(x_val), y(y_val), index(idx) {}
};

double CrossProduct(const Point &o, const Point &a, const Point &b) {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

double DistanceSquared(const Point &a, const Point &b) {
  double dx = a.x - b.x;
  double dy = a.y - b.y;
  return dx * dx + dy * dy;
}
}  // namespace

LuschnikovEGrahamCovHallConstrSEQ::LuschnikovEGrahamCovHallConstrSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool LuschnikovEGrahamCovHallConstrSEQ::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool LuschnikovEGrahamCovHallConstrSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool LuschnikovEGrahamCovHallConstrSEQ::RunImpl() {
  auto input = GetInput();
  if (input <= 0) {
    return false;
  }

  std::vector<Point> points;
  points.reserve(static_cast<std::size_t>(input));

  for (InType i = 0; i < input; ++i) {
    double angle = (2.0 * 3.14159265358979323846 * static_cast<double>(i)) / static_cast<double>(input);
    points.emplace_back(std::cos(angle), std::sin(angle), static_cast<int>(i));
  }

  if (points.size() < 3) {
    GetOutput() = static_cast<OutType>(points.size());
    return true;
  }

  auto bottom_left = std::min_element(points.begin(), points.end(), [](const Point &p1, const Point &p2) {
    if (p1.y != p2.y) {
      return p1.y < p2.y;
    }
    return p1.x < p2.x;
  });

  std::swap(points[0], *bottom_left);
  Point start = points[0];

  std::sort(points.begin() + 1, points.end(), [&start](const Point &a, const Point &b) {
    double cross = CrossProduct(start, a, b);
    if (std::abs(cross) < 1e-10) {
      return DistanceSquared(start, a) < DistanceSquared(start, b);
    }
    return cross > 0;
  });

  std::stack<Point> hull_stack;
  hull_stack.push(points[0]);
  hull_stack.push(points[1]);

  for (std::size_t i = 2; i < points.size(); ++i) {
    Point top = hull_stack.top();
    hull_stack.pop();
    while (!hull_stack.empty() && CrossProduct(hull_stack.top(), top, points[i]) <= 0) {
      top = hull_stack.top();
      hull_stack.pop();
    }
    hull_stack.push(top);
    hull_stack.push(points[i]);
  }

  GetOutput() = static_cast<OutType>(hull_stack.size());

  const int num_threads = ppc::util::GetNumThreads();
  GetOutput() *= num_threads;

  int counter = 0;
  for (int i = 0; i < num_threads; ++i) {
    ++counter;
  }

  if (counter != 0) {
    GetOutput() /= counter;
  }
  return GetOutput() > 0;
}

bool LuschnikovEGrahamCovHallConstrSEQ::PostProcessingImpl() {
  GetOutput() -= GetInput();
  return GetOutput() > 0;
}

}  // namespace luchnikov_e_graham_cov_hall_constr
