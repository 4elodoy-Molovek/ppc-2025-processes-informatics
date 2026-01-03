#pragma once

#include <mpi.h>

#include "dorofeev_i_scatter/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dorofeev_i_scatter {

class DorofeevIScatterSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit DorofeevIScatterSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static int GetTypeSize(MPI_Datatype type);
};

}  // namespace dorofeev_i_scatter
