#include <cassert>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "nlohmann/json.hpp"
#include "pic.hpp"
#include "semiLagrangian.hpp"
#include "utils.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;

// creates a directory for files, and copies the json file there
fs::path working_dir_handling(json &data, char *json_path) {
    std::cout << "Creating the working directory" << std::endl;

    // Getting a non-existing default directory
    std::string base_dir = "simulation";
    std::string default_dir = base_dir;
    int i = 0;
    while (fs::exists(default_dir)) {
        i++;
        default_dir = base_dir + "_" + std::to_string(i);
    }

    // Loading the user's choice
    fs::path dir = data.value("dir", default_dir);
    if (fs::exists(dir)) {
        std::cout << "[WARN] " << dir << " already exists, switching to "
                  << default_dir << std::endl;
        dir = default_dir;
    }

    // creating the dir, and copying the json file there
    fs::create_directory(dir);
    fs::create_directory(dir / "data");
    fs::copy(json_path, dir / fs::path(json_path).filename());

    return dir;
}

int main(int argc, char **argv) {
    // Check args
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <param.json>\n";
        return EXIT_FAILURE;
    }

    // Load json file
    std::ifstream inputf(argv[1]);
    json data = json::parse(inputf);

    // Handles the directory handling
    fs::path work_dir = working_dir_handling(data, argv[1]);

    // Open log file as write and append
    // Handles a default case
    // std::string log_file_path;
    fs::path log_file_path;
    if (data.contains("log_file") &&
        data["log_file"].type() == json::value_t::string)
        log_file_path = work_dir / (fs::path)data["log_file"];
    else
        log_file_path = work_dir / "log.txt";

    std::ofstream log_file(log_file_path, std::ios::out | std::ios::app);
    if (!log_file.is_open()) {
        std::cerr << "Could not open the log file: " << log_file_path << "\n";
        return EXIT_FAILURE;
    }

    fs::path metrics_file_path;
    if (data.contains("metrics_file") &&
        data["metrics_file"].type() == json::value_t::string)
        metrics_file_path = work_dir / (fs::path)data["metrics_file"];
    else
        metrics_file_path = work_dir / "metrics.csv";

    std::ofstream metrics_file(metrics_file_path, std::ios::out | std::ios::app);
    if (!metrics_file.is_open()) {
        std::cerr << "Could not open the metrics file: " << metrics_file_path << "\n";
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
    std::cout << "OpenMP available: OMP_NUM_THREADS=" << omp_get_max_threads()
              << std::endl;
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
        ret = solver_semi_lagrangian(data, log_file, work_dir);
    } else if (data["solver"] == "pic") {
        ret = solver_pic(data, log_file, metrics_file, work_dir);
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