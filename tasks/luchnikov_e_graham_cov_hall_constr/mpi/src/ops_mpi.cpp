#include "luchnikov_e_graham_cov_hall_constr/mpi/include/ops_mpi.hpp"

#include <mpi.h>

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
constexpr double kPi = 3.14159265358979323846;
constexpr double kEpsilon = 1e-10;
constexpr std::size_t kPointDataSize = 3;
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
void PackPoints(const std::vector<Point> &points, std::vector<double> &buffer) {
  buffer.resize(points.size() * kPointDataSize);
  for (std::size_t i = 0; i < points.size(); ++i) {
    buffer[i * kPointDataSize] = points[i].x;
    buffer[i * kPointDataSize + 1] = points[i].y;
    buffer[i * kPointDataSize + 2] = static_cast<double>(points[i].index);
  }
}
std::vector<Point> UnpackPoints(const std::vector<double> &buffer) {
  std::vector<Point> points;
  std::size_t count = buffer.size() / kPointDataSize;
  points.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    Point p;
    p.x = buffer[i * kPointDataSize];
    p.y = buffer[i * kPointDataSize + 1];
    p.index = static_cast<int>(buffer[i * kPointDataSize + 2]);
    points.push_back(p);
  }
  return points;
}
std::vector<Point> GrahamScan(std::vector<Point> points) {
  if (points.size() < 3) {
    return points;
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
    if (std::abs(cross) < kEpsilon) {
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
  std::vector<Point> result;
  result.reserve(hull_stack.size());
  while (!hull_stack.empty()) {
    result.push_back(hull_stack.top());
    hull_stack.pop();
  }
  return result;
}
}  // namespace
LuschnikovEGrahamCovHallConstrMPI::LuschnikovEGrahamCovHallConstrMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}
bool LuschnikovEGrahamCovHallConstrMPI::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}
bool LuschnikovEGrahamCovHallConstrMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}
bool LuschnikovEGrahamCovHallConstrMPI::RunImpl() {
  auto input = GetInput();
  if (input <= 0) {
    return false;
  }
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::vector<Point> all_points;
  if (rank == 0) {
    all_points.reserve(static_cast<std::size_t>(input));
    for (InType i = 0; i < input; ++i) {
      double angle = (2.0 * kPi * static_cast<double>(i)) / static_cast<double>(input);
      all_points.emplace_back(std::cos(angle), std::sin(angle), static_cast<int>(i));
    }
  }
  int points_per_proc = static_cast<int>(input) / size;
  int remainder = static_cast<int>(input) % size;
  std::vector<int> send_counts(static_cast<std::size_t>(size), 0);
  std::vector<int> displs(static_cast<std::size_t>(size), 0);
  for (int i = 0; i < size; ++i) {
    send_counts[static_cast<std::size_t>(i)] = (i < remainder) ? (points_per_proc + 1) : points_per_proc;
    if (i > 0) {
      displs[static_cast<std::size_t>(i)] =
          displs[static_cast<std::size_t>(i - 1)] + send_counts[static_cast<std::size_t>(i - 1)];
    }
  }
  int local_count = send_counts[static_cast<std::size_t>(rank)];
  std::vector<Point> local_points;
  local_points.reserve(static_cast<std::size_t>(local_count));
  if (rank == 0) {
    for (int i = 0; i < local_count; ++i) {
      local_points.push_back(all_points[static_cast<std::size_t>(i)]);
    }
    for (int i = 1; i < size; ++i) {
      int start_idx = displs[static_cast<std::size_t>(i)];
      int count = send_counts[static_cast<std::size_t>(i)];
      MPI_Send(&count, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      if (count > 0) {
        std::vector<double> buffer;
        std::vector<Point> temp_points;
        temp_points.reserve(static_cast<std::size_t>(count));
        for (int j = 0; j < count; ++j) {
          temp_points.push_back(all_points[static_cast<std::size_t>(start_idx + j)]);
        }
        PackPoints(temp_points, buffer);
        MPI_Send(buffer.data(), static_cast<int>(buffer.size()), MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
      }
    }
  } else {
    int recv_count = 0;
    MPI_Recv(&recv_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (recv_count > 0) {
      std::vector<double> buffer(static_cast<std::size_t>(recv_count) * kPointDataSize);
      MPI_Recv(buffer.data(), static_cast<int>(buffer.size()), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      local_points = UnpackPoints(buffer);
    }
  }
  auto local_hull = GrahamScan(local_points);
  std::vector<Point> global_hull;
  if (rank == 0) {
    global_hull = std::move(local_hull);
    for (int i = 1; i < size; ++i) {
      int recv_count = 0;
      MPI_Recv(&recv_count, 1, MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (recv_count > 0) {
        std::vector<double> buffer(static_cast<std::size_t>(recv_count) * kPointDataSize);
        MPI_Recv(buffer.data(), static_cast<int>(buffer.size()), MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        auto recv_points = UnpackPoints(buffer);
        global_hull.insert(global_hull.end(), recv_points.begin(), recv_points.end());
      }
    }
    if (!global_hull.empty()) {
      global_hull = GrahamScan(global_hull);
    }
    GetOutput() = static_cast<OutType>(global_hull.size());
  } else {
    int local_hull_size = static_cast<int>(local_hull.size());
    MPI_Send(&local_hull_size, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    if (local_hull_size > 0) {
      std::vector<double> buffer;
      PackPoints(local_hull, buffer);
      MPI_Send(buffer.data(), static_cast<int>(buffer.size()), MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  return GetOutput() > 0;
}
bool LuschnikovEGrahamCovHallConstrMPI::PostProcessingImpl() {
  return GetOutput() > 0;
}
}  // namespace luchnikov_e_graham_cov_hall_constr
