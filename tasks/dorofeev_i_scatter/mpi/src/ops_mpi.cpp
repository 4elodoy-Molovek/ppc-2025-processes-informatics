#include "dorofeev_i_scatter/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "dorofeev_i_scatter/common/include/common.hpp"

namespace dorofeev_i_scatter {

DorofeevIScatterMPI::DorofeevIScatterMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = nullptr;
}

bool DorofeevIScatterMPI::ValidationImpl() {
  auto [sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm] = GetInput();
  int size = 0;
  MPI_Comm_size(comm, &size);
  return sendcount >= 0 && recvcount >= 0 && root >= 0 && root < size;
}

bool DorofeevIScatterMPI::PreProcessingImpl() {
  return true;
}

bool DorofeevIScatterMPI::RunImpl() {
  auto [sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm] = GetInput();

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int type_size = GetTypeSize(sendtype);

  // Tree-based scatter: each process receives a chunk and forwards remaining data
  if (rank == root) {
    // Root keeps its own data
    std::copy_n(static_cast<const char *>(sendbuf) + (static_cast<ptrdiff_t>(rank * sendcount * type_size)),
                sendcount * type_size, static_cast<char *>(recvbuf));

    // Send remaining data to first child
    int first_child = 1;
    if (first_child < size) {
      int remaining_elements = (size - 1) * sendcount;
      MPI_Send(static_cast<const char *>(sendbuf) + (static_cast<ptrdiff_t>(first_child * sendcount * type_size)),
               remaining_elements, sendtype, first_child, 0, comm);
    }

  } else {
    // Non-root processes: receive from previous process in the chain
    int sender = rank - 1;

    // Calculate how many elements we expect to receive
    int remaining_processes = size - rank;
    int elements_to_recv = remaining_processes * sendcount;

    std::vector<char> recv_buffer(static_cast<size_t>(elements_to_recv * type_size));

    MPI_Recv(recv_buffer.data(), elements_to_recv, recvtype, sender, 0, comm, MPI_STATUS_IGNORE);

    // Keep our data
    std::copy_n(recv_buffer.data(), sendcount * type_size, static_cast<char *>(recvbuf));

    // Send remaining data to next process
    int next_process = rank + 1;
    if (next_process < size) {
      int remaining_elements = (remaining_processes - 1) * sendcount;
      MPI_Send(recv_buffer.data() + (static_cast<ptrdiff_t>(sendcount * type_size)), remaining_elements, sendtype,
               next_process, 0, comm);
    }
  }

  GetOutput() = recvbuf;
  return true;
}

bool DorofeevIScatterMPI::PostProcessingImpl() {
  return true;
}

int DorofeevIScatterMPI::GetTypeSize(MPI_Datatype type) {
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
