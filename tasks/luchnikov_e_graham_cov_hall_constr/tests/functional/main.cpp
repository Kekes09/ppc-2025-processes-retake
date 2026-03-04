#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "luchnikov_e_graham_cov_hall_constr/common/include/common.hpp"
#include "luchnikov_e_graham_cov_hall_constr/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_graham_cov_hall_constr/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
namespace luchnikov_e_graham_cov_hall_constr {
class LuschnikovEGrahamCovHallConstrFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }
  bool CheckTestOutputData(OutType &output_data) final {
    return (input_data_ == output_data);
  }
  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};
namespace {
TEST_P(LuschnikovEGrahamCovHallConstrFuncTestsProcesses, MatmulFromPic) {
  ExecuteTest(GetParam());
}
// SEQ тесты
TEST(LuschnikovEGrahamCovHallConstrSEQTest, TrianglePoints) {
  InType input = 3;
  LuschnikovEGrahamCovHallConstrSEQ task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}
TEST(LuschnikovEGrahamCovHallConstrSEQTest, SquarePoints) {
  InType input = 4;
  LuschnikovEGrahamCovHallConstrSEQ task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}
TEST(LuschnikovEGrahamCovHallConstrSEQTest, PentagonPoints) {
  InType input = 5;
  LuschnikovEGrahamCovHallConstrSEQ task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}
TEST(LuschnikovEGrahamCovHallConstrSEQTest, HexagonPoints) {
  InType input = 6;
  LuschnikovEGrahamCovHallConstrSEQ task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}
TEST(LuschnikovEGrahamCovHallConstrSEQTest, HeptagonPoints) {
  InType input = 7;
  LuschnikovEGrahamCovHallConstrSEQ task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}
TEST(LuschnikovEGrahamCovHallConstrSEQTest, OctagonPoints) {
  InType input = 8;
  LuschnikovEGrahamCovHallConstrSEQ task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}
TEST(LuschnikovEGrahamCovHallConstrSEQTest, NonagonPoints) {
  InType input = 9;
  LuschnikovEGrahamCovHallConstrSEQ task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}
TEST(LuschnikovEGrahamCovHallConstrSEQTest, DecagonPoints) {
  InType input = 10;
  LuschnikovEGrahamCovHallConstrSEQ task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}
TEST(LuschnikovEGrahamCovHallConstrSEQTest, CirclePoints) {
  InType input = 50;
  LuschnikovEGrahamCovHallConstrSEQ task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}
TEST(LuschnikovEGrahamCovHallConstrSEQTest, SmallInput) {
  InType input = 2;
  LuschnikovEGrahamCovHallConstrSEQ task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}
TEST(LuschnikovEGrahamCovHallConstrSEQTest, InvalidInput) {
  InType input = -5;
  LuschnikovEGrahamCovHallConstrSEQ task(input);
  EXPECT_FALSE(task.Validation());
}
// [ИСПРАВЛЕНО] Восстановлены MPI тесты
TEST(LuschnikovEGrahamCovHallConstrMPITest, TrianglePoints) {
  InType input = 3;
  LuschnikovEGrahamCovHallConstrMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}
TEST(LuschnikovEGrahamCovHallConstrMPITest, SquarePoints) {
  InType input = 4;
  LuschnikovEGrahamCovHallConstrMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}
TEST(LuschnikovEGrahamCovHallConstrMPITest, PentagonPoints) {
  InType input = 5;
  LuschnikovEGrahamCovHallConstrMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}
TEST(LuschnikovEGrahamCovHallConstrMPITest, HexagonPoints) {
  InType input = 6;
  LuschnikovEGrahamCovHallConstrMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}
TEST(LuschnikovEGrahamCovHallConstrMPITest, HeptagonPoints) {
  InType input = 7;
  LuschnikovEGrahamCovHallConstrMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}
TEST(LuschnikovEGrahamCovHallConstrMPITest, OctagonPoints) {
  InType input = 8;
  LuschnikovEGrahamCovHallConstrMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}
TEST(LuschnikovEGrahamCovHallConstrMPITest, NonagonPoints) {
  InType input = 9;
  LuschnikovEGrahamCovHallConstrMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}
TEST(LuschnikovEGrahamCovHallConstrMPITest, DecagonPoints) {
  InType input = 10;
  LuschnikovEGrahamCovHallConstrMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}
TEST(LuschnikovEGrahamCovHallConstrMPITest, CirclePoints) {
  InType input = 50;
  LuschnikovEGrahamCovHallConstrMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}
TEST(LuschnikovEGrahamCovHallConstrMPITest, SmallInput) {
  InType input = 2;
  LuschnikovEGrahamCovHallConstrMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}
TEST(LuschnikovEGrahamCovHallConstrMPITest, InvalidInput) {
  InType input = -5;
  LuschnikovEGrahamCovHallConstrMPI task(input);
  EXPECT_FALSE(task.Validation());
}
const std::array<TestType, 10> kTestParam = {std::make_tuple(3, "triangle"),  std::make_tuple(4, "square"),
                                             std::make_tuple(5, "pentagon"),  std::make_tuple(6, "hexagon"),
                                             std::make_tuple(7, "heptagon"),  std::make_tuple(8, "octagon"),
                                             std::make_tuple(9, "nonagon"),   std::make_tuple(10, "decagon"),
                                             std::make_tuple(20, "icosagon"), std::make_tuple(100, "large_circle")};
const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<LuschnikovEGrahamCovHallConstrMPI, InType>(
                                               kTestParam, PPC_SETTINGS_luchnikov_e_graham_cov_hall_constr),
                                           ppc::util::AddFuncTask<LuschnikovEGrahamCovHallConstrSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_luchnikov_e_graham_cov_hall_constr));
const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = LuschnikovEGrahamCovHallConstrFuncTestsProcesses::PrintFuncTestName<
    LuschnikovEGrahamCovHallConstrFuncTestsProcesses>;
INSTANTIATE_TEST_SUITE_P(GrahamScanTests, LuschnikovEGrahamCovHallConstrFuncTestsProcesses, kGtestValues,
                         kPerfTestName);
}  // namespace
}  // namespace luchnikov_e_graham_cov_hall_constr
