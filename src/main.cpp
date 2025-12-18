#include <iostream>
#include <cassert>
#include <fstream>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "nlohmann/json.hpp"
using json = nlohmann::json;

int main(int argc, char **argv)
{
#ifdef _OPENMP
    std::cout << "OpenMP available: OMP_NUM_THREADS=" << omp_get_max_threads() << "\n";
#else
    std::cout << "OpenMP not available.\n";
#endif

#ifdef NDEBUG
    // code has been configured with "cmake -DCMAKE_BUILD_TYPE=Release .."
    std::cout << "code built in RELEASE mode.\n";
#else
    // code has been configured with "cmake .."
    std::cout << "code built in DEBUG mode.\n";
#endif
    
    // assert(2==3); // make program crash in Debug mode, but not in Release mode

    // read json filename as first argument
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <param.json>\n";
        return EXIT_FAILURE;
    }
    // read input data from json file given as argument

    std::ifstream inputf(argv[1]);
    json data = json::parse(inputf);

    // print input data to screen
    std::cout << argv[1] << ":\n" << data.dump(4) << std::endl;

    return EXIT_SUCCESS;
}
