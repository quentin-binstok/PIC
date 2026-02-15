
#ifndef __SOLVER_SEMI_LAGRANGIAN__
#define __SOLVER_SEMI_LAGRANGIAN__

#include "data.hpp"
#include "nlohmann/json.hpp"

#include <fstream>

using json = nlohmann::json;

void gauss_seidel(scalar_field *p, scalar_field *vx, scalar_field *vy,
                  scalar_field *div, float dx, float dt, float tol,
                  std::ofstream &log_file);

int solver_semi_lagrangian(json &data, std::ofstream &log_file);

#endif
