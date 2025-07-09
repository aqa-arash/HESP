#include <omp.h>
#include <iostream>
int main() {
  int num_devices = omp_get_num_devices();
  printf("Number of devices: %d\n", num_devices);
  #pragma omp target
  {
    if (omp_is_initial_device()) printf("Running on CPU\n"); else printf("Running on Intel GPU\n");
  }
  return 0;
}
