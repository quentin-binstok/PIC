#ifndef __SOLVER_THERMAL__
#define __SOLVER_THERMAL__

#include "data.hpp"
#include "nlohmann/json.hpp"
#include <vector>

using json = nlohmann::json;

enum THERM_BC_TYPE { DIRICHLET_THERM, NEUMANN_THERM };

// 0 = left, 1 = right, 2 = top, 3 = bottom
typedef struct _therm_bc {
    std::vector<THERM_BC_TYPE> type;
    std::vector<float> val;
} therm_bc;

int particles_temp_to_grid(particle_field *particles, scalar_field *T,
                           scalar_field *kern_sum_T, std::ofstream &log_file);

int grid_temp_to_particles(particle_field *particles, scalar_field *T,
                           std::ofstream &log_file);

void apply_thermal_eq(scalar_field *T, scalar_field *T_temp, therm_bc *bcs,
                      float dt, float c, float rho, float k, float tol,
                      int max_iter, std::ofstream &log_file);

void build_thermal_bc(therm_bc *bcs, json &data, std::ofstream &log_file);

float interp_temp(float x, float y, float dx, scalar_field *T);
#endif
