#ifndef __SOLVER_PARTICLES__
#define __SOLVER_PARTICLES__

#include "data.hpp"
#include "nlohmann/json.hpp"
#include "poisson.hpp"
#include "utils.hpp"
#include "conditions.hpp"

// ─────────────────────────────────────────────
//  Function declarations
// ─────────────────────────────────────────────

/**
 * @brief Advects a single particle with a 3-stage 3rd-order RK scheme.
 *        x and y are read as the initial position and overwritten with the new one.
 */
void advect_single_particle(float *x, float *y,
                             scalar_field *vx, scalar_field *vy,
                             float dt, std::ofstream &log_file);

/**
 * @brief Advects all particles in the field (OpenMP parallel).
 */
int advect(particle_field *particles,
           scalar_field *vx, scalar_field *vy,
           float dt, std::ofstream &log_file);

/**
 * @brief Linear interpolation kernel used when transferring particle quantities to the grid.
 */
float kernel(float r);

/**
 * @brief Gradient of the linear interpolation kernel.
 */
float kernel_grad(float x);


/**
 * @brief Applies gravitational acceleration to all particle velocities.
 */
void apply_gravity(particle_field *particles, float g, float dt);

/**
 * @brief Places particles in every LIQUID cell of the domain,
 *        initialising their positions and velocities from the grid fields.
 */
void initialize_particles(particle_field *particles,
                           scalar_field *dom,
                           scalar_field *vx, scalar_field *vy,
                           int density, std::ofstream &log_file);

/**
 * @brief Removes particle at index p (swap-with-last idiom, O(1)).
 */
void remove_particle(particle_field *particles, int p);

/**
 * @brief Recomputes per-cell density, removes out-of-bounds and solid particles.
 */
int check_particles(particle_field *particles,
                    scalar_field *dom,
                    Metrics &m,
                    std::vector<int> &density,
                    std::ofstream &log_file);

/**
 * @brief Fills cell (i, j) with new particles up to imposed_density.
 */
void fill_cell(int i, int j,
               particle_field *particles,
               scalar_field *vx, scalar_field *vy, scalar_field *dom,
               int imposed_density, std::vector<int> &density,
               float dt, RNG &rng, Metrics &m, std::ofstream &log_file);

/**
 * @brief Iterates over the whole domain:
 *        - replenishes DIRICHLET inflow cells
 *        - refills underpopulated LIQUID cells (when refill == true)
 *        - converts depopulated LIQUID cells to AIR
 *        - converts populated AIR cells back to LIQUID
 */
void refill_domain(particle_field *particles,
                   scalar_field *dom,
                   scalar_field *vx, scalar_field *vy,
                   std::vector<int> &density,
                   int particle_density,
                   bool refill, float creation_rate,
                   float dt, RNG &rng, Metrics &m, std::ofstream &log_file);

void compute_C(particle_field *particles, scalar_field *vx,
                   scalar_field *vy, int p, Metrics &m);

#endif