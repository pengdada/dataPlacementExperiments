#include <cstdio>
#include <vector>
#include <string>
#include <thrust/device_vector.h>

#include <cupti_profiler.h>

/* 
 * @Abdullah, Naming the configuration used
 * This is just for testing purpose, so coding a little crude
 */
char* SetUpConfigurationName(int argc, char* argv[])


template<typename T>
__global__ void kernel(T begin, int size) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if(thread_id < size)
    *(begin + thread_id) += 1;
}

template<typename T>
__global__ void kernel2(T begin, int size) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if(thread_id < size)
    *(begin + thread_id) += 2;
}

template<typename T>
void call_kernel(T& arg) {
  kernel<<<1, 100>>>(thrust::raw_pointer_cast(&arg[0]), arg.size());
}

template<typename T>
void call_kernel2(T& arg) {
  kernel2<<<1, 50>>>(thrust::raw_pointer_cast(&arg[0]), arg.size());
}

int main() {
  using namespace std;
  //using namespace thrust;

  vector<string> event_names {
                              /*"global_store",
                              "shared_store"*/
                              /*"active_warps",*/
                              /*"gst_inst_32bit"*//*,
                              "active_cycles",
                              "threads_launched",
                              "branch"*/
                             };
  vector<string> metric_names {
                               /*"flop_count_dp",
                               "flop_count_sp",
                               "inst_executed",
                               "gst_transactions",
                               "gld_transactions",
                               "shared_efficiency",
                               "l2_tex_read_hit_rate",
                               "l2_atomic_transactions",
                               "l2_tex_write_hit_rate",
                               "l2_read_transactions",
                               "l2_utilization",
                               "l2_atomic_throughput",
                               "l2_tex_write_transactions",
                               "l2_read_throughput",
                               "l2_tex_read_transactions",
                               "l2_write_transactions",
                               "stall_memory_throttle"*/
  	"shared_load_transactions_per_request","shared_store_transactions_per_request", "local_load_transactions_per_request", "local_store_transactions_per_request", "gld_transactions_per_request", "gst_transactions_per_request", "shared_store_transactions", "shared_load_transactions", "local_load_transactions", "local_store_transactions", "gld_transactions", "gst_transactions", "sysmem_read_transactions", "sysmem_write_transactions", "l2_read_transactions", "l2_write_transactions", "dram_read_transactions", "dram_write_transactions", "global_hit_rate", "local_hit_rate", "gld_requested_throughput", "gst_requested_throughput", "gld_throughput", "gst_throughput", "local_memory_overhead", "tex_cache_hit_rate", "l2_tex_read_hit_rate", "l2_tex_write_hit_rate", "dram_read_throughput", "dram_write_throughput", "tex_cache_throughput", "l2_tex_read_throughput", "l2_tex_write_throughput", "l2_read_throughput", "l2_write_throughput", "sysmem_read_throughput", "sysmem_write_throughput", "local_load_throughput", "local_store_throughput", "shared_load_throughput", "shared_store_throughput", "gld_efficiency", "gst_efficiency", "tex_cache_transactions", "dram_utilization", "sysmem_utilization", "stall_memory_dependency", "stall_texture", "stall_constant_memory_dependency", "shared_efficiency", "inst_compute_ld_st", "ldst_issued", "ldst_executed", "atomic_transactions", "atomic_transactions_per_request", "l2_atomic_throughput", "l2_atomic_transactions", "l2_tex_read_transactions", "stall_memory_throttle", "l2_tex_write_transactions", "shared_utilization", "l2_utilization", "tex_utilization", "ldst_fu_utilization", "tex_fu_utilization", "sysmem_read_utilization", "sysmem_write_utilization", "pcie_total_data_transmitted", "pcie_total_data_received", "inst_executed_global_loads", "inst_executed_local_loads", "inst_executed_shared_loads", "inst_executed_surface_loads", "inst_executed_global_stores", "inst_executed_local_stores", "inst_executed_shared_stores", "inst_executed_surface_stores", "inst_executed_tex_ops", "dram_read_bytes", "dram_write_bytes", "global_load_requests", "local_load_requests", "surface_load_requests", "global_store_requests", "local_store_requests", "surface_store_requests", "l2_global_load_bytes", "l2_local_load_bytes", "l2_surface_load_bytes", "l2_global_atomic_store_bytes", "l2_local_global_store_bytes", "l2_surface_store_bytes", "sysmem_read_bytes", "sysmem_write_bytes", "l2_tex_hit_rate", "texture_load_requests"
                              };

  constexpr int N = 100;
  thrust::device_vector<double> data(N, 0);

  //cupti_profiler::profiler profiler(vector<string>{}, metric_names);

  // XXX: Disabling all metrics seems to change the values
  // of some events. Not sure if this is correct behavior.
  //cupti_profiler::profiler profiler(event_names, vector<string>{});
	for(int i=0; i<10; ++i) {
	    call_kernel(data);
	    cudaDeviceSynchronize();
	    call_kernel2(data);
	    cudaDeviceSynchronize();
	}
  cupti_profiler::profiler profiler(event_names, metric_names);
  // Get #passes required to compute all metrics and events
  const int passes = profiler.get_passes();
  printf("Passes: %d\n", passes);




  profiler.start();
  cudaDeviceSynchronize();
  for(int i=0; i<passes; ++i) {
    call_kernel(data);
    cudaDeviceSynchronize();
    call_kernel2(data);
    cudaDeviceSynchronize();
  }
  profiler.stop();

  printf("Event Trace\n");
  profiler.print_event_values(std::cerr);
  printf("Metric Trace\n");
  profiler.print_metric_values_to_file("output.csv", "default");

  auto names = profiler.get_kernel_names();
  for(auto name: names) {
    printf("%s\n", name.c_str());
  }

  thrust::host_vector<float> h_data(data);

  /*printf("\n");
  for(int i = 0; i < 10; ++i) {
    printf("%.2lf ", h_data[i]);
  }*/
  printf("\n");
  return 0;
}
