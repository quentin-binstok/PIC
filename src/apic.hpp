#ifndef __SOLVER_PIC__
#define __SOLVER_PIC__

#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>

using json = nlohmann::json;
namespace fs = std::filesystem;

int solver_pic(json &data, std::ofstream &log_file, std::ofstream &metrics_file, fs::path work_dir);

#endif
