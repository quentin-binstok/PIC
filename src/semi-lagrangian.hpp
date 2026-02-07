#ifndef __SOLVER_SEMI_LAGRANGIAN__
#define __SOLVER_SEMI_LAGRANGIAN__

#include "nlohmann/json.hpp"
#include <fstream>

using json = nlohmann::json;

/*
 @brief the semi lagrangian solver
 @param data: the whole json
 @param log_file: the log file
*/
int solver_semi_lagrangian(json &data, std::ofstream &log_file);

#endif