#pragma once

#include "luchnikov_e_graham_cov_hall_constr/common/include/common.hpp"
#include "task/include/task.hpp"

namespace luchnikov_e_graham_cov_hall_constr {

class LuschnikovEGrahamCovHallConstrMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit LuschnikovEGrahamCovHallConstrMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace luchnikov_e_graham_cov_hall_constr
