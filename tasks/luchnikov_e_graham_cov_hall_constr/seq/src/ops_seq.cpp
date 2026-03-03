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
constexpr double kEpsilon = 1e-10;
constexpr std::size_t kMinHullPoints = 3;
struct Point {
  double x;
  double y;
  int index;
  Point() : x(0.0), y(0.0), index(0) {}
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
bool ComparePointsByYThenX(const Point &p1, const Point &p2) {
  if (p1.y != p2.y) {
    return p1.y < p2.y;
  }
  return p1.x < p2.x;
}
bool CompareByPolarAngle(const Point &start, const Point &a, const Point &b) {
  double cross = CrossProduct(start, a, b);
  if (std::abs(cross) < kEpsilon) {
    return DistanceSquared(start, a) < DistanceSquared(start, b);
  }
  return cross > 0;
}
std::size_t BuildConvexHull(std::vector<Point> &points) {
  if (points.size() < kMinHullPoints) {
    return points.size();
  }
  auto bottom_left = std::min_element(points.begin(), points.end(), ComparePointsByYThenX);
  std::swap(points[0], *bottom_left);
  Point start = points[0];
  std::sort(points.begin() + 1, points.end(),
            [&start](const Point &a, const Point &b) { return CompareByPolarAngle(start, a, b); });
  std::stack<Point> hull_stack;
  hull_stack.push(points[0]);
  hull_stack.push(points[1]);
  for (std::size_t i = kMinHullPoints - 1; i < points.size(); ++i) {
    Point top = hull_stack.top();
    hull_stack.pop();
    while (!hull_stack.empty() && CrossProduct(hull_stack.top(), top, points[i]) <= 0) {
      top = hull_stack.top();
      hull_stack.pop();
    }
    hull_stack.push(top);
    hull_stack.push(points[i]);
  }
  return hull_stack.size();
}
std::vector<Point> GenerateConvexPoints(InType count) {
  std::vector<Point> points;
  points.reserve(static_cast<std::size_t>(count));
  for (InType i = 0; i < count; ++i) {
    double x = static_cast<double>(i);
    double y = x * x;
    points.emplace_back(x, y, static_cast<int>(i));
  }
  return points;
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
  std::vector<Point> points = GenerateConvexPoints(input);
  GetOutput() = static_cast<OutType>(BuildConvexHull(points));
  return GetOutput() > 0;
}
bool LuschnikovEGrahamCovHallConstrSEQ::PostProcessingImpl() {
  return GetOutput() > 0;
}
}  // namespace luchnikov_e_graham_cov_hall_constr
