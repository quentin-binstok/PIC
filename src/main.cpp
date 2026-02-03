#include "test.hpp"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cassert>
#include <fstream>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "nlohmann/json.hpp"
#include "data.hpp"
#include "utils.hpp"
using json = nlohmann::json;

int main(int argc, char **argv)
{
    // Check args
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <param.json>\n";
        return EXIT_FAILURE;
    }

    // Load json file
    std::ifstream inputf(argv[1]);
    json data = json::parse(inputf);

    // Open log file as write and append
    std::ofstream log_file (data["log_file"], std::ios::out | std::ios::app);
    if (!log_file.is_open()) {
        std::cerr << "Could not open the log file: " << data["log_file"] << "\n";
        return EXIT_FAILURE;
    }

    // Starting log file
    time_t timestamp;
    time(&timestamp);
    LOG_INFO(log_file, "STARTING SIMULATION");
    LOG_INFO(log_file, "Date: " << ctime(&timestamp));
 
    // OMP log
#ifdef _OPENMP
    LOG_INFO(log_file, "OpenMP available: OMP_NUM_THREADS=" << omp_get_max_threads());
#else
    LOG_INFO(log_file, "OpenMP not available.");
#endif

    // Debug log
#ifdef NDEBUG
    // code has been configured with "cmake -DCMAKE_BUILD_TYPE=Release .."
    LOG_INFO(log_file, "Code built in RELEASE mode.");
#else
    // code has been configured with "cmake .."
    LOG_INFO(log_file, "Code built in DEBUG mode.");
#endif

    test_vtp(log_file);
    
    return EXIT_SUCCESS;
}
