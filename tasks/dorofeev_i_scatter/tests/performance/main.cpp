#include <gtest/gtest.h>
#include <mpi.h>

#include <cstdlib>
#include <tuple>
#include <vector>

#include "dorofeev_i_scatter/common/include/common.hpp"
#include "dorofeev_i_scatter/mpi/include/ops_mpi.hpp"
#include "dorofeev_i_scatter/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace dorofeev_i_scatter {

class DorofeevIScatterPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    // For performance testing, use a large array of doubles
    send_data_.resize(100000000, 1.0);      // 100M elements
    recv_data_.resize(100000000 / 4, 0.0);  // Assume 4 processes, each gets 25M

    // Create input tuple: (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm)
    input_ = std::make_tuple(send_data_.data(), 25000000, MPI_DOUBLE, recv_data_.data(), 25000000, MPI_DOUBLE, 0,
                             MPI_COMM_WORLD);
  }

  InType GetTestInputData() override {
    return input_;
  }

  bool CheckTestOutputData(OutType &output_data) override {
    // For performance tests, just check that output is not null
    return output_data != nullptr;
  }

 private:
  std::vector<double> send_data_;
  std::vector<double> recv_data_;
  InType input_;
};

TEST_P(DorofeevIScatterPerfTests, ScatterPerf) {
  ExecuteTest(GetParam());
}

const auto kPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, DorofeevIScatterMPI, DorofeevIScatterSEQ>(PPC_SETTINGS_dorofeev_i_scatter);

INSTANTIATE_TEST_SUITE_P(ScatterPerf, DorofeevIScatterPerfTests, ppc::util::TupleToGTestValues(kPerfTasks),
                         DorofeevIScatterPerfTests::CustomPerfTestName);

}  // namespace dorofeev_i_scatter
