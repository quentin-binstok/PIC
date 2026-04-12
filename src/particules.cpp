#include "data.hpp"
#include "nlohmann/json.hpp"
#include "poisson.hpp"
#include "utils.hpp"
#include "conditions.hpp"



/*
 @brief advects a single particle with the APIC scheme
 @param x, y: must contain the original position, will be overwritten
*/
void advect_single_particle(float *x, float *y, scalar_field *vx,
                                   scalar_field *vy, float dt,
                                   std::ofstream &log_file) {

    // Three-stage third-order RK scheme
    float init_x = *x;
    float init_y = *y;

    float k_1x, k_1y;
    get_speed(&k_1x, &k_1y, init_x, init_y, vx, vy, log_file);

    float x2 = init_x + 0.5 * dt * k_1x;
    float y2 = init_y + 0.5 * dt * k_1y;
    float k_2x, k_2y;
    get_speed(&k_2x, &k_2y, x2, y2, vx, vy, log_file);

    float x3 = init_x + 0.75 * dt * k_2x;
    float y3 = init_y + 0.75 * dt * k_2y;
    float k_3x, k_3y;
    get_speed(&k_3x, &k_3y, x3, y3, vx, vy, log_file);

    float x_new = init_x + (2.0f / 9.0f) * dt * k_1x +
                  (3.0f / 9.0f) * dt * k_2x + (4.0f / 9.0f) * dt * k_3x;
    float y_new = init_y + (2.0f / 9.0f) * dt * k_1y +
                  (3.0f / 9.0f) * dt * k_2y + (4.0f / 9.0f) * dt * k_3y;

    *x = x_new;
    *y = y_new;
}

/*
 @brief avects the particles based on their velocity, using the APIC scheme
*/
int advect(particle_field *particles, scalar_field *vx,
                      scalar_field *vy, float dt, std::ofstream &log_file) {
    LOG_INFO(log_file, "Advecting particles");

#pragma omp parallel for
    for (int k = 0; k < particles->N; k++) {
        // Three-stage third-order RK scheme
        float x = particles->xyz[2 * k];
        float y = particles->xyz[2 * k + 1];

        float k_1x, k_1y;
        get_speed(&k_1x, &k_1y, x, y, vx, vy, log_file);

        float x2 = x + 0.5 * dt * k_1x;
        float y2 = y + 0.5 * dt * k_1y;
        float k_2x, k_2y;
        get_speed(&k_2x, &k_2y, x2, y2, vx, vy, log_file);

        float x3 = x + 0.75 * dt * k_2x;
        float y3 = y + 0.75 * dt * k_2y;
        float k_3x, k_3y;
        get_speed(&k_3x, &k_3y, x3, y3, vx, vy, log_file);

        float x_new = x + (2.0f / 9.0f) * dt * k_1x +
                      (3.0f / 9.0f) * dt * k_2x + (4.0f / 9.0f) * dt * k_3x;
        float y_new = y + (2.0f / 9.0f) * dt * k_1y +
                      (3.0f / 9.0f) * dt * k_2y + (4.0f / 9.0f) * dt * k_3y;

        particles->xyz[2 * k] = x_new;
        particles->xyz[2 * k + 1] = y_new;
    }

    return EXIT_SUCCESS;
}

/*
 @brief the kernel used to bring back particule fields on the grid
*/
float kernel(float r) {
    if (0 <= r && r <= 1)
        return 1 - r;
    if (-1 <= r && r <= 0)
        return 1 + r;
    return 0;
}

// Kinematic application of gravity
void apply_gravity(particle_field *particles, float g, float dt) {
#pragma omp parallel for
    for (int k = 0; k < particles->N; k++) {
        particles->velocity[2 * k + 1] -= g * dt;
    }
}

/*
 @brief Initializes the particles on the grid
*/
void initialize_particles(particle_field *particles,
                                     scalar_field *dom, scalar_field *vx,
                                     scalar_field *vy, int density,
                                     std::ofstream &log_file) {
    // Initialization of stuff
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-0.5 * dom->dx, 0.5 * dom->dx);
    int nx = dom->nx, ny = dom->ny;
    float dx = dom->dx;

    int nb_not_fluid_cases = 0;
    int current_id = 0;

    particles->xyz.resize(2 * density * nx * ny);
    particles->velocity.resize(2 * density * nx * ny);

#pragma omp parallel for collapse(2)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            float cell = GET(dom, i, j);
            // Only populate liquid cells
            if (cell != SOLID && cell != AIR) {
                for (int k = 0; k < density; k++) {
                    float x, y;
#pragma omp critical
                    x = i * dx + dist(gen);
#pragma omp critical
                    y = j * dx + dist(gen);
                    particles->xyz[2 * current_id] = x;
                    particles->xyz[2 * current_id + 1] = y;

                    // To have speed initialization
                    float v_x, v_y;
                    get_speed(&v_x, &v_y, x, y, vx, vy, log_file);

                    particles->velocity[2 * current_id] = v_x;
                    particles->velocity[2 * current_id + 1] = v_y;

#pragma omp atomic
                    current_id++;
                }

            } else {
#pragma omp atomic
                nb_not_fluid_cases++;
            }
        }
    }

    // After the loop:
    particles->N = current_id;
    particles->next_id = current_id; // next to assign = current_id
    particles->xyz.resize(2 * current_id);
    particles->velocity.resize(2 * current_id);
    particles->B.resize(4*current_id);

    // Fill ids properly
    particles->id.resize(current_id);
    std::iota(particles->id.begin(), particles->id.end(), 0);
}

// Removes a single particle of index p (not ID!)
void remove_particle(particle_field *particles, int p) {
    int last = particles->N - 1;

    // Early exit if removing the last particle (no swap needed)
    if (p != last) {
        std::swap(particles->xyz[2 * p], particles->xyz[2 * last]);
        std::swap(particles->xyz[2 * p + 1], particles->xyz[2 * last + 1]);

        std::swap(particles->velocity[2 * p], particles->velocity[2 * last]);
        std::swap(particles->velocity[2 * p + 1], particles->velocity[2 * last + 1]);

        std::swap(particles->B[4 * p], particles->B[4 * last]);
        std::swap(particles->B[4 * p + 1], particles->B[4 * last + 1]);
        std::swap(particles->B[4 * p + 2], particles->B[4 * last + 2]);
        std::swap(particles->B[4 * p + 3], particles->B[4 * last + 3]);


        std::swap(particles->id[p], particles->id[last]);
    }

    particles->xyz.pop_back();
    particles->xyz.pop_back();
    particles->velocity.pop_back();
    particles->velocity.pop_back();
    particles->B.pop_back();
    particles->B.pop_back();
    particles->B.pop_back();
    particles->B.pop_back();
    particles->id.pop_back();

    particles->N--;
}

/*
 @brief computes the density and removes particles in invalid places
*/
int check_particles(particle_field *particles, scalar_field *dom, Metrics& m,
                           std::vector<int> &density, std::ofstream &log_file) {
    LOG_INFO(log_file, "Updating particles")

    int nx = dom->nx;
    int ny = dom->ny;
    float dx = dom->dx;

    std::fill(density.begin(), density.end(), 0);
    m.particle_count = 0;

    // Remove invalid particles
    int p = 0;
    while (p < particles->N) {

        float x = particles->xyz[2 * p];
        float y = particles->xyz[2 * p + 1];

        // exact cell
        int i = (int)(x / dx + 0.5f);
        int j = (int)(y / dx + 0.5f);

        // Check if the particle is out of bounds
        if (i < 0 || j < 0 || i >= nx || j >= ny) {
            remove_particle(particles, p);
            m.particle_count++;
            continue;
        }

        // Check if the particle is in a solid cell
        float cell = GET(dom, i, j);
        if (cell == SOLID) {
            remove_particle(particles, p);
            m.particle_count++;
            continue;
        }

        density[j * nx + i]++;
        p++;
    }

    return EXIT_SUCCESS;
}

/*
 @brief fills the cell (i,j) with particles until imposed_density
*/
void fill_cell(int i, int j, particle_field *particles, scalar_field *vx,
                      scalar_field *vy, scalar_field *dom, int imposed_density,
                      std::vector<int> &density, float dt, RNG &rng,
                      std::ofstream &log_file) {
    int nx = vx->nx;
    int ny = vx->ny;
    float dx = vx->dx;

    int idx = j * nx + i;
    int cell_density = density[idx];

    int to_add = std::max(0, imposed_density - cell_density);

    for (int k = 0; k < to_add; ++k) {

        // Spawn position (jittered inside cell)
        float x = i * dx + rng.jitter(rng.gen);
        float y = j * dx + rng.jitter(rng.gen);

        // Get velocity at position
        float vx_p, vy_p;
        get_speed(&vx_p, &vy_p, x, y, vx, vy, log_file);

        // Random birth time integration
        float tau = rng.birth(rng.gen);
        float remaining = dt - tau;

        advect_single_particle(&x, &y, vx, vy, remaining, log_file);

        // Compute new cell index (faster than floor)
        int fi = (int)(x / dx + 0.5f);
        int fj = (int)(y / dx + 0.5f);

        // Bounds + solid check (merged for branch efficiency)
        if (fi < 0 || fj < 0 || fi >= nx || fj >= ny)
            continue;

        if (GET(dom, fi, fj) == SOLID)
            continue;

        // Add particle
        particles->xyz.push_back(x);
        particles->xyz.push_back(y);

        particles->velocity.push_back(vx_p);
        particles->velocity.push_back(vy_p);

        particles->B.push_back(0);
        particles->B.push_back(0);
        particles->B.push_back(0);
        particles->B.push_back(0);

        particles->id.push_back(particles->next_id++);

        // Optional: remove if you switch to size()
        particles->N++;

        // Update density
        density[fj * nx + fi]++;
    }
}

/*
 @brief refills the domain, change cell types
*/
void refill_domain(particle_field *particles, scalar_field *dom,
                          scalar_field *vx, scalar_field *vy,
                          std::vector<int> &density, int particle_density,
                          bool refill, float creation_rate, float dt, RNG &rng,
                          std::ofstream &log_file) {
    LOG_INFO(log_file, "Refilling domain");

    int nx = dom->nx;
    int ny = dom->ny;

#pragma omp parallel for collapse(2)
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {

            int idx = j * nx + i;

            float cell_type = GET(dom, i, j);
            int cell_density = density[idx];

            // --- DIRICHLET: inflow ---
            if (cell_type == DIRICHLET) {

                float n = dt * creation_rate;
                int target_number = (int)n;

#pragma omp critical
                fill_cell(i, j, particles, vx, vy, dom, target_number, density,
                          dt, rng, log_file);
            }

            // --- LIQUID cells ---
            else if (cell_type == LIQUID) {

                if (refill && cell_density < particle_density) {
#pragma omp critical
                    fill_cell(i, j, particles, vx, vy, dom, particle_density, density,
                              dt, rng, log_file);
                } else if (cell_density < 1) {
                    SET(dom, i, j, AIR);
                }
            }

            // --- AIR cells (convert back to liquid if needed) ---
            else if (cell_type == AIR && i > 0 && i < nx - 1 && j > 0 &&
                     j < ny - 1) {

                if (cell_density > 0) {
                    SET(dom, i, j, LIQUID);
                }
            }
        }
    }
}