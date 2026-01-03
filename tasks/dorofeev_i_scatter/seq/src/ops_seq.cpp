#include "dorofeev_i_scatter/seq/include/ops_seq.hpp"

#include <mpi.h>

#include <algorithm>

#include "dorofeev_i_scatter/common/include/common.hpp"

namespace dorofeev_i_scatter {

DorofeevIScatterSEQ::DorofeevIScatterSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = nullptr;
}

bool DorofeevIScatterSEQ::ValidationImpl() {
  auto [sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm] = GetInput();
  return sendcount >= 0 && recvcount >= 0 && root == 0;  // Sequential always has root 0
}

bool DorofeevIScatterSEQ::PreProcessingImpl() {
  return true;
}

bool DorofeevIScatterSEQ::RunImpl() {
  auto [sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm] = GetInput();

  // In sequential version, just copy from sendbuf to recvbuf
  // Since there's only one process, it gets all the data
  int type_size = GetTypeSize(sendtype);
  std::copy_n(static_cast<const char *>(sendbuf), sendcount * type_size, static_cast<char *>(recvbuf));

  GetOutput() = recvbuf;
  return true;
}

bool DorofeevIScatterSEQ::PostProcessingImpl() {
  return true;
}

int DorofeevIScatterSEQ::GetTypeSize(MPI_Datatype type) {
  if (type == MPI_INT) {
    return sizeof(int);
  }
  if (type == MPI_FLOAT) {
    return sizeof(float);
  }
  if (type == MPI_DOUBLE) {
    return sizeof(double);
  }
  return 1;  // fallback
}

}  // namespace dorofeev_i_scatter
