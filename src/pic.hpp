
#ifndef __SOLVER_PIC__
#define __SOLVER_PIC__

#include "nlohmann/json.hpp"
#include <fstream>

using json = nlohmann::json;

int solver_pic(json &data, std::ofstream &log_file);

#endif
