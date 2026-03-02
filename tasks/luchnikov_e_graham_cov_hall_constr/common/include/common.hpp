#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace luchnikov_e_graham_cov_hall_constr {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace luchnikov_e_graham_cov_hall_constr
