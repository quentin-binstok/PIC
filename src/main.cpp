#include <cstdlib>
#include <ctime>
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
        std::cerr << "Could not open the log file: " << data["log_file"] << std::endl;
        return EXIT_FAILURE;
    }

    // Starting log file
    time_t timestamp;
    time(&timestamp);
    log_file << "STARTING SIMULATION" << std::endl;
    log_file << "[INFO]" << ctime(&timestamp) << std::endl;
 
    // OMP log
#ifdef _OPENMP
    log_file << "[INFO] OpenMP available: OMP_NUM_THREADS=" << omp_get_max_threads() << "\n";
#else
    log_file << "[INFO] OpenMP not available.\n";
#endif

    // Debug log
#ifdef NDEBUG
    // code has been configured with "cmake -DCMAKE_BUILD_TYPE=Release .."
    log_file << "[INFO] code built in RELEASE mode.\n";
#else
    // code has been configured with "cmake .."
    log_file << "[INFO] code built in DEBUG mode.\n";
#endif
    
    return EXIT_SUCCESS;
}
