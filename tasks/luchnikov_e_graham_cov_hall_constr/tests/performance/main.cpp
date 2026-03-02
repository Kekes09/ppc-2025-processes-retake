#include <gtest/gtest.h>

#include "luchnikov_e_graham_cov_hall_constr/common/include/common.hpp"
#include "luchnikov_e_graham_cov_hall_constr/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_graham_cov_hall_constr/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace luchnikov_e_graham_cov_hall_constr {

class LuschnikovEGrahamCovHallConstrPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 1000;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LuschnikovEGrahamCovHallConstrPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LuschnikovEGrahamCovHallConstrMPI, LuschnikovEGrahamCovHallConstrSEQ>(
        PPC_SETTINGS_luchnikov_e_graham_cov_hall_constr);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LuschnikovEGrahamCovHallConstrPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LuschnikovEGrahamCovHallConstrPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace luchnikov_e_graham_cov_hall_constr
