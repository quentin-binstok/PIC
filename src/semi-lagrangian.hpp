#ifndef __SOLVER_SEMI_LAGRANGIAN__
#define __SOLVER_SEMI_LAGRANGIAN__

#include "nlohmann/json.hpp"
#include <fstream>

using json = nlohmann::json;

int solver_semi_lagrangian(json& data, std::ofstream& log_file);

#endif