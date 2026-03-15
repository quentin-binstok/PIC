
#include "test.hpp"
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "nlohmann/json.hpp"
#include "data.hpp"
#include "pic.hpp"
#include "semiLagrangian.hpp"
#include "utils.hpp"
using json = nlohmann::json;
int main(int argc, char **argv) {
    // Check args
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <param.json>\n";
        return EXIT_FAILURE;
    }
    // Load json file
    std::ifstream inputf(argv[1]);
    json data = json::parse(inputf);
    // Open log file as write and append
    // Handles a default case
    std::string log_file_path;
    if (data.contains("log_file") &&
        data["log_file"].type() == json::value_t::string)
        log_file_path = data["log_file"];
    else
        log_file_path = "log.txt";
    std::ofstream log_file(log_file_path, std::ios::out | std::ios::app);
    if (!log_file.is_open()) {
        std::cerr << "Could not open the log file: " << log_file_path << "\n";
        return EXIT_FAILURE;
    }
    // Starting log file, with starting time
    time_t timestamp;
    time(&timestamp);
    LOG_INFO(log_file, "STARTING SIMULATION");
    LOG_INFO(log_file, "Date: " << ctime(&timestamp));
    // OMP log
#ifdef _OPENMP
    LOG_INFO(log_file,
             "OpenMP available: OMP_NUM_THREADS=" << omp_get_max_threads());
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
    // Launches the solver, using a common error handling
    int ret = 0;
    if (data["solver"] == "semi-lagrangian") {
        ret = solver_semi_lagrangian(data, log_file);
    } else if (data["solver"] == "pic") {
        ret = solver_pic(data, log_file);
    } else {
        LOG_ERR(log_file, "The specified solver is not supported.");
        LOG_ERR(log_file, "Exiting.")
        return EXIT_FAILURE;
    }
    if (ret == EXIT_FAILURE) {
        LOG_ERR(log_file, "An error occured in the solver.");
        return EXIT_FAILURE;
    }
    // no forgetting that
    log_file.close();
    return EXIT_SUCCESS;
}