#ifndef __SOLVER_APIC__
#define __SOLVER_APIC__

#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>

using json = nlohmann::json;
namespace fs = std::filesystem;

int solver_apic(json &data, std::ofstream &log_file, std::ofstream &metrics_file, fs::path work_dir);

#endif
