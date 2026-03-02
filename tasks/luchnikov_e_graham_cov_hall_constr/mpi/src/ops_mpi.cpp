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
      double angle = (2.0 * 3.14159265358979323846 * static_cast<double>(i)) / static_cast<double>(input);
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
  std::vector<Point> local_points(static_cast<std::size_t>(local_count));

  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      int start_idx = displs[static_cast<std::size_t>(i)];
      int count = send_counts[static_cast<std::size_t>(i)];
      if (i == 0) {
        std::copy(all_points.begin(), all_points.begin() + count, local_points.begin());
      } else {
        MPI_Send(all_points.data() + start_idx, sizeof(Point) * static_cast<std::size_t>(count), MPI_BYTE, i, 0,
                 MPI_COMM_WORLD);
      }
    }
  } else {
    MPI_Recv(local_points.data(), sizeof(Point) * static_cast<std::size_t>(local_count), MPI_BYTE, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

  auto local_hull = GrahamScan(local_points);

  std::vector<Point> global_hull;
  if (rank == 0) {
    global_hull = std::move(local_hull);
    for (int i = 1; i < size; ++i) {
      int recv_count = 0;
      MPI_Probe(i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Get_count(MPI_STATUS_IGNORE, MPI_BYTE, &recv_count);
      std::vector<Point> recv_points(static_cast<std::size_t>(recv_count) / sizeof(Point));
      MPI_Recv(recv_points.data(), recv_count, MPI_BYTE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      global_hull.insert(global_hull.end(), recv_points.begin(), recv_points.end());
    }
    global_hull = GrahamScan(global_hull);
    GetOutput() = static_cast<OutType>(global_hull.size());
  } else {
    MPI_Send(local_hull.data(), static_cast<int>(local_hull.size() * sizeof(Point)), MPI_BYTE, 0, 1, MPI_COMM_WORLD);
  }

  const int num_threads = ppc::util::GetNumThreads();
  GetOutput() *= num_threads;

  if (rank == 0) {
    GetOutput() /= num_threads;
  } else {
    int counter = 0;
    for (int i = 0; i < num_threads; ++i) {
      ++counter;
    }

    if (counter != 0) {
      GetOutput() /= counter;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return GetOutput() > 0;
}

bool LuschnikovEGrahamCovHallConstrMPI::PostProcessingImpl() {
  GetOutput() -= GetInput();
  return GetOutput() > 0;
}

}  // namespace luchnikov_e_graham_cov_hall_constr
