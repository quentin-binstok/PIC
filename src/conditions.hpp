
#ifndef __SOLVER_CONDITIONS__
#define __SOLVER_CONDITIONS__
#include <fstream>
#include "data.hpp"
#include "nlohmann/json.hpp"
using json = nlohmann::json;
int apply_initial_condition(scalar_field *field, json &data,
                            std::string condition_name,
                            std::ofstream &log_file);
#endif