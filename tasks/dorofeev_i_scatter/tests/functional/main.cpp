#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "dorofeev_i_scatter/common/include/common.hpp"
#include "dorofeev_i_scatter/mpi/include/ops_mpi.hpp"
#include "dorofeev_i_scatter/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace dorofeev_i_scatter {

class DorofeevIScatterFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 protected:
  void SetUp() override {
    const auto &params = std::get<2>(GetParam());
    int count = std::get<0>(params);
    int root = std::get<1>(params);
    MPI_Datatype type = std::get<2>(params);

    // Create test data based on type
    if (type == MPI_DOUBLE) {
      send_data_double_.resize(static_cast<size_t>(count) * 4, 0.0);  // 4 processes
      recv_data_double_.resize(count, 0.0);
      for (size_t i = 0; i < send_data_double_.size(); ++i) {
        send_data_double_[i] = static_cast<double>(i);
      }

      input_ = std::make_tuple(send_data_double_.data(), count, type, recv_data_double_.data(), count, type, root,
                               MPI_COMM_WORLD);
    } else if (type == MPI_INT) {
      send_data_int_.resize(static_cast<size_t>(count) * 4, 0);
      recv_data_int_.resize(count, 0);
      for (size_t i = 0; i < send_data_int_.size(); ++i) {
        send_data_int_[i] = static_cast<int>(i);
      }

      input_ =
          std::make_tuple(send_data_int_.data(), count, type, recv_data_int_.data(), count, type, root, MPI_COMM_WORLD);
    } else if (type == MPI_FLOAT) {
      send_data_float_.resize(static_cast<size_t>(count) * 4, 0.0F);
      recv_data_float_.resize(count, 0.0F);
      for (size_t i = 0; i < send_data_float_.size(); ++i) {
        send_data_float_[i] = static_cast<float>(i);
      }

      input_ = std::make_tuple(send_data_float_.data(), count, type, recv_data_float_.data(), count, type, root,
                               MPI_COMM_WORLD);
    }
  }

  InType GetTestInputData() override {
    return input_;
  }

  bool CheckTestOutputData(OutType &out) override {  // NOLINT(readability-function-cognitive-complexity)
    if (out == nullptr) {
      return false;
    }

    // Get task name to determine if this is MPI or sequential test
    const auto &task_name = std::get<1>(GetParam());
    bool is_sequential = task_name.find("seq") != std::string::npos;

    // Validate that we received the correct data
    const auto &params = std::get<2>(GetParam());
    int count = std::get<0>(params);
    // int root = std::get<1>(params);  // Not used in validation
    MPI_Datatype type = std::get<2>(params);

    int rank = 0;
    int size = 1;
    if (!is_sequential) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);
    }

    // Check that each process has the correct data
    if (type == MPI_DOUBLE) {
      const auto *received = static_cast<const double *>(out);
      if (is_sequential) {
        // Sequential: process gets all data (count * size elements)
        for (int i = 0; i < count * size; ++i) {
          auto expected = static_cast<double>(i);
          if (received[i] != expected) {
            return false;
          }
        }
      } else {
        // MPI: each process gets count elements starting from rank * count
        for (int i = 0; i < count; ++i) {
          auto expected = static_cast<double>((rank * count) + i);
          if (received[i] != expected) {
            return false;
          }
        }
      }
    } else if (type == MPI_INT) {
      const int *received = static_cast<const int *>(out);
      if (is_sequential) {
        // Sequential: process gets all data (count * size elements)
        for (int i = 0; i < count * size; ++i) {
          int expected = i;
          if (received[i] != expected) {
            return false;
          }
        }
      } else {
        // MPI: each process gets count elements starting from rank * count
        for (int i = 0; i < count; ++i) {
          int expected = (rank * count) + i;
          if (received[i] != expected) {
            return false;
          }
        }
      }
    } else if (type == MPI_FLOAT) {
      const auto *received = static_cast<const float *>(out);
      if (is_sequential) {
        // Sequential: process gets all data (count * size elements)
        for (int i = 0; i < count * size; ++i) {
          auto expected = static_cast<float>(i);
          if (received[i] != expected) {
            return false;
          }
        }
      } else {
        // MPI: each process gets count elements starting from rank * count
        for (int i = 0; i < count; ++i) {
          auto expected = static_cast<float>((rank * count) + i);
          if (received[i] != expected) {
            return false;
          }
        }
      }
    }

    return true;
  }

 public:
  static std::string PrintTestParam(const TestType &param) {
    int count = std::get<0>(param);
    int root = std::get<1>(param);
    MPI_Datatype type = std::get<2>(param);
    std::string type_str;
    if (type == MPI_DOUBLE) {
      type_str = "double";
    } else if (type == MPI_INT) {
      type_str = "int";
    } else {
      type_str = "float";
    }
    return std::to_string(count) + "_" + std::to_string(root) + "_" + type_str;
  }

 private:
  std::vector<double> send_data_double_;
  std::vector<double> recv_data_double_;
  std::vector<int> send_data_int_;
  std::vector<int> recv_data_int_;
  std::vector<float> send_data_float_;
  std::vector<float> recv_data_float_;
  InType input_;
};

TEST_P(DorofeevIScatterFuncTests, ScatterCorrectness) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParams = {
    std::make_tuple(4, 0, MPI_DOUBLE),  // count=4, root=0, type=double
    std::make_tuple(3, 0, MPI_INT),     // count=3, root=0, type=int
    std::make_tuple(2, 0, MPI_FLOAT),   // count=2, root=0, type=float
};

const auto kTasks =
    std::tuple_cat(ppc::util::AddFuncTask<DorofeevIScatterMPI, InType>(kTestParams, PPC_SETTINGS_dorofeev_i_scatter),
                   ppc::util::AddFuncTask<DorofeevIScatterSEQ, InType>(kTestParams, PPC_SETTINGS_dorofeev_i_scatter));

INSTANTIATE_TEST_SUITE_P(ScatterTests, DorofeevIScatterFuncTests, ppc::util::ExpandToValues(kTasks),
                         DorofeevIScatterFuncTests::PrintFuncTestName<DorofeevIScatterFuncTests>);

}  // namespace dorofeev_i_scatter
