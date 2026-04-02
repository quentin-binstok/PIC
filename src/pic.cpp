#include "conditions.hpp"
#include "data.hpp"
#include "nlohmann/json.hpp"
#include "poisson.hpp"
#include "utils.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ostream>
#include <random>

using json = nlohmann::json;
namespace fs = std::filesystem;

/*
 @brief advects a single particle with the PIC scheme
 @param x, y: must contain the original position, will be overwritten
*/
inline void advect_single_particle(float *x, float *y, scalar_field *vx,
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
 @brief avects the particles based on their velocity, using the PIC scheme
*/
inline int advect_pic(particle_field *particles, scalar_field *vx,
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
inline float kernel(float r) {
    if (0 <= r && r <= 1)
        return 1 - r;
    if (-1 <= r && r <= 0)
        return 1 + r;
    return 0;
}

/*
 @brief brings the particule fields to the grid
*/
inline int particles_speed_to_grid(particle_field *particles, scalar_field *vx,
                                   scalar_field *vy, scalar_field *kern_sum_vx,
                                   scalar_field *kern_sum_vy,
                                   std::ofstream &log_file) {
    LOG_INFO(log_file, "Transferring the speed of particles to the grid");

    int nx = vy->nx, ny = vx->ny;
    float dx = vx->dx;

// Zero things out
#pragma omp parallel for collapse(2)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            SET(vx, i, j, 0);
            SET(vy, i, j, 0);
            SET(kern_sum_vx, i, j, 0);
            SET(kern_sum_vy, i, j, 0);
        }
    }

    // Gets the speeds, the kernel weights
#pragma omp parallel for
    for (int k = 0; k < particles->N; k++) {
        float x = particles->xyz[2 * k];
        float y = particles->xyz[2 * k + 1];
        float u = particles->velocity[2 * k];
        float v = particles->velocity[2 * k + 1];

        // --- vx stencil: staggered in x, collocated in y ---
        int i_vx = std::max(0, (int)(x / dx - 0.5)); // matches get_speed
        int j_vx = std::max(0, (int)(y / dx));
        for (int j = j_vx; j < std::min(ny, j_vx + 2); j++) {
            for (int i = i_vx; i < std::min(nx, i_vx + 2); i++) {
                float kern = kernel((x - i * dx - 0.5f * dx) / dx) *
                             kernel((y - j * dx) / dx);
#pragma omp atomic
                vx->values[j * nx + i] += u * kern;
#pragma omp atomic
                kern_sum_vx->values[j * nx + i] += kern;
            }
        }

        // --- vy stencil: collocated in x, staggered in y ---
        int i_vy = std::max(0, (int)(x / dx));
        int j_vy = std::max(0, (int)(y / dx - 0.5)); // matches get_speed
        for (int j = j_vy; j < std::min(ny, j_vy + 2); j++) {
            for (int i = i_vy; i < std::min(nx, i_vy + 2); i++) {
                float kern = kernel((x - i * dx) / dx) *
                             kernel((y - j * dx - 0.5f * dx) / dx);
#pragma omp atomic
                vy->values[j * nx + i] += v * kern;
#pragma omp atomic
                kern_sum_vy->values[j * nx + i] += kern;
            }
        }
    }

    // Divide by the total kernel weight
#pragma omp parallel for collapse(2)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            float kern_x = GET(kern_sum_vx, i, j);
            float kern_y = GET(kern_sum_vy, i, j);
            if (kern_x != 0) {
                SET(vx, i, j, GET(vx, i, j) / kern_x);
            }
            if (kern_y != 0)
                SET(vy, i, j, GET(vy, i, j) / kern_y);
        }
    }

    return EXIT_SUCCESS;
}

/*
 @brief interpolates the grid fields to the particles
*/
inline int grid_speed_to_particles(particle_field *particles, scalar_field *vx,
                                   scalar_field *vy, scalar_field *vx_old,
                                   scalar_field *vy_old, float flip_param,
                                   std::ofstream &log_file) {

#pragma omp parallel for
    for (int k = 0; k < particles->N; k++) {
        float x = particles->xyz[2 * k];
        float y = particles->xyz[2 * k + 1];

        // gets the speeds on the old and new grids
        float v_x, v_y, v_x_old, v_y_old;
        get_speed(&v_x, &v_y, x, y, vx, vy, log_file);
        get_speed(&v_x_old, &v_y_old, x, y, vx_old, vy_old, log_file);

        particles->velocity[2 * k] =
            (1 - flip_param) * v_x +
            flip_param * (particles->velocity[2 * k] + v_x - v_x_old);
        particles->velocity[2 * k + 1] =
            (1 - flip_param) * v_y +
            flip_param * (particles->velocity[2 * k + 1] + v_y - v_y_old);
    }

    return EXIT_SUCCESS;
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
inline void initialize_particles_pic(particle_field *particles,
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
        std::swap(particles->velocity[2 * p + 1],
                  particles->velocity[2 * last + 1]);

        std::swap(particles->id[p], particles->id[last]);
    }

    particles->xyz.pop_back();
    particles->xyz.pop_back();
    particles->velocity.pop_back();
    particles->velocity.pop_back();
    particles->id.pop_back();

    particles->N--;
}

/*
 @brief computes the density and removes particles in invalid places
*/
inline int check_particles(particle_field *particles, scalar_field *dom,
                           std::vector<int> &density, std::ofstream &log_file) {
    LOG_INFO(log_file, "Updating particles")

    int nx = dom->nx;
    int ny = dom->ny;
    float dx = dom->dx;

    std::fill(density.begin(), density.end(), 0);

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
            continue;
        }

        // Check if the particle is in a solid cell
        float cell = GET(dom, i, j);
        if (cell == SOLID) {
            remove_particle(particles, p);
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
inline void fill_cell(int i, int j, particle_field *particles, scalar_field *vx,
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
inline void refill_domain(particle_field *particles, scalar_field *dom,
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
                    fill_cell(i, j, particles, vx, vy, dom, particle_density,
                              density, dt, rng, log_file);
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

/*
 @brief the PIC/FLIP
 @param data: the whole json
 @param log_file: the log file
*/
int solver_pic(json &data, std::ofstream &log_file, fs::path work_dir) {
    LOG_INFO(log_file, "Starting the PIC/FLIP solver");

    auto t0 = std::chrono::high_resolution_clock::now();

    if (check_params(data, log_file)) {
        LOG_ERR(log_file, "Problem checking the input parameters")
        return EXIT_FAILURE;
    }

    // Getting base params
    unsigned int nx = data["grid"][0], ny = data["grid"][1];
    float dx = data["space_steps"];
    int sampling_rate = data["sampling_rate"];
    float dt = data.value("delta_t", 0.1);
    unsigned int nt = data.value("nt", 10);
    float rho = data.value("rho", 1000);
    float tol = data.value("tol", 1e-5);
    int max_iter = data.value("max_iter", 1e5);
    int particle_density = data.value("particle_density", 8);
    bool refill = data.value("refill", false);
    float flip_param = data.value("flip", 0.0f);
    LOG_INFO(log_file, "FLIP percentage is " << flip_param * 100);
    bool gravity = data.value("gravity", false);
    float g = data.value("g", 9.81);
    RNG rng(dx, dt);

    // Computing the creation rate
    float speed_x = 0.0f;
    float speed_y = 0.0f;
    for (auto &bc : data["bc"]) {
        if (bc.contains("speed_x")) {
            if (speed_x == 0) {
                speed_x = bc["speed_x"].get<float>();
                continue;
            } else {
                speed_x /= 2;
                speed_x += bc["speed_x"].get<float>() / 2;
            }
        }
        if (bc.contains("speed_y")) {
            if (speed_y == 0) {
                speed_y = bc["speed_y"].get<float>();
                continue;
            } else {
                speed_y /= 2;
                speed_y += bc["speed_y"].get<float>() / 2;
            }
        }
    }

    float mean_speed = 0.0f;
    if (speed_x && speed_y)
        mean_speed = (speed_x + speed_y) / 2;
    else if (speed_x)
        mean_speed = speed_x;
    else
        mean_speed = speed_y;
    int creation_rate = particle_density * mean_speed / dx;

    // Initialising the fields
    scalar_field *vx = scalar_field_init("vx", nx, ny, 0.5, 0, dx, log_file);
    scalar_field *vy = scalar_field_init("vy", nx, ny, 0, 0.5, dx, log_file);
    scalar_field *p = scalar_field_init("p", nx, ny, 0, 0, dx, log_file);
    scalar_field *div = scalar_field_init("div", nx, ny, 0, 0, dx, log_file);
    scalar_field *dom = scalar_field_init("dom", nx, ny, 0, 0, dx, log_file);
    scalar_field *temp_vx = scalar_field_copy(vx, log_file);
    scalar_field *temp_vy = scalar_field_copy(vy, log_file);
    scalar_field *temp_p = scalar_field_copy(p, log_file);

    scalar_field *kern_sum_vx =
        scalar_field_init("kern_sum_vx", nx, ny, 0, 0, dx, log_file);
    scalar_field *kern_sum_vy =
        scalar_field_init("kern_sum_vy", nx, ny, 0, 0, dx, log_file);

    if (!vx || !vy || !p || !div || !dom || !temp_vx || !temp_vy || !temp_p ||
        !kern_sum_vx || !kern_sum_vy) {
        LOG_ERR(log_file, "An error occured initializing fields.");
        return EXIT_FAILURE;
    }

    std::vector<float> speed_condition;

    // Applying the initial conditions
    initialize_speed(vx, dom, data, "ic_vx", log_file);
    initialize_speed(vy, dom, data, "ic_vy", log_file);
    boundary_condition(vx, vy, dom, speed_condition, data, "bc", log_file);
    initialize_domain(dom, data, "ic_cell", log_file);
    create_circle(dom, "ic_cylinders", data, log_file);

    // Initializing the particles
    particle_field *particles = particle_field_init_2D(
        "particles", nx * ny * particle_density, log_file);
    initialize_particles_pic(particles, dom, vx, vy, data["particle_density"],
                             log_file);

    // Manifests
    write_manifest_vtk("particles", dt, nt, sampling_rate, 1, 1, log_file,
                       work_dir);
    write_particles_vtp(particles, 0, 0, 2, log_file, work_dir);

    write_manifest_vtk(vx->name, dt, nt, sampling_rate, 1, 0, log_file,
                       work_dir);
    write_manifest_vtk(vy->name, dt, nt, sampling_rate, 1, 0, log_file,
                       work_dir);
    write_manifest_vtk(p->name, dt, nt, sampling_rate, 1, 0, log_file,
                       work_dir);
    write_manifest_vtk(div->name, dt, nt, sampling_rate, 1, 0, log_file,
                       work_dir);
    write_manifest_vtk(dom->name, dt, nt, sampling_rate, 1, 0, log_file,
                       work_dir);

    // Initial state
    write_scalar_vtk(vx, 0, 0, log_file, work_dir);
    write_scalar_vtk(vy, 0, 0, log_file, work_dir);
    write_scalar_vtk(p, 0, 0, log_file, work_dir);
    write_scalar_vtk(div, 0, 0, log_file, work_dir);
    write_scalar_vtk(dom, 0, 0, log_file, work_dir);

    std::vector<int> density(nx * ny, 0);

    // Main time loop
    // bool inverted = false;
    bool first_loop = true;
    std::cout << "Starting simulation" << std::endl;
    for (unsigned int i = 1; i < nt; i++) {
        // Logging and printing stuff
        log_file << "\n";
        LOG_INFO(log_file, "Starting time loop " << i << " out of " << nt);
        if (i % (nt / 10) == 0) {
            auto tnow = std::chrono::high_resolution_clock::now();
            double seconds = std::chrono::duration<double>(tnow - t0).count();
            double sec_per_iter = seconds / i;
            std::cout << "\rRemaining computation time: "
                      << (nt - i) * sec_per_iter << "s\t";
            std::cout << "Iteration " << i << "/" << nt;
            std::flush(std::cout);
        }

        if (gravity)
            apply_gravity(particles, g, dt);

        particles_speed_to_grid(particles, vx, vy, kern_sum_vx, kern_sum_vy,
                                log_file);

        divergence(vx, vy, div, dom, speed_condition, log_file);

        if (data["iteration_algo"] == "Jacobi")
            jacobi(p, temp_p, div, vx, vy, dom, tol, dt, rho, max_iter,
                   first_loop, log_file);
        else if (data["iteration_algo"] == "SOR")
            sor(p, div, vx, vy, dom, tol, dt, rho, max_iter, log_file);
        else {
            LOG_ERR(log_file, "Iteration algorithm not supported");
            return EXIT_FAILURE;
        }

        // Needed for FLIP
        std::memcpy(temp_vx->values, vx->values, nx * ny * sizeof(float));
        std::memcpy(temp_vy->values, vy->values, nx * ny * sizeof(float));

        project_velocity(p, vx, vy, dom, dx, dt, rho, log_file,
                         speed_condition);

        // This is to be able to save it. It serves no purpose in the
        // algorithm
        divergence(vx, vy, div, dom, speed_condition, log_file);

        grid_speed_to_particles(particles, vx, vy, temp_vx, temp_vy, flip_param,
                                log_file);

        // save files
        if (sampling_rate && !(i % sampling_rate)) {
            write_scalar_vtk(vx, i, 0, log_file, work_dir);
            write_scalar_vtk(vy, i, 0, log_file, work_dir);
            write_scalar_vtk(p, i, 0, log_file, work_dir);
            write_scalar_vtk(div, i, 0, log_file, work_dir);
            write_scalar_vtk(dom, i, 0, log_file, work_dir);

            write_particles_vtp(particles, i, 0, 2, log_file, work_dir);
        }

        advect_pic(particles, vx, vy, dt, log_file);

        std::fill(density.begin(), density.end(), 0);
        check_particles(particles, dom, density, log_file);
        refill_domain(particles, dom, vx, vy, density, particle_density, refill,
                      creation_rate, dt, rng, log_file);

        first_loop = false;
    }

    // As we're not using objects, we need this
    scalar_field_free(vx, log_file);
    scalar_field_free(vy, log_file);
    scalar_field_free(p, log_file);
    scalar_field_free(div, log_file);
    scalar_field_free(dom, log_file);
    scalar_field_free(temp_vx, log_file);
    scalar_field_free(temp_vy, log_file);
    scalar_field_free(temp_p, log_file);

    scalar_field_free(kern_sum_vx, log_file);
    scalar_field_free(kern_sum_vy, log_file);

    free(particles);

    auto t1 = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(t1 - t0).count();
    LOG_INFO(log_file, "Total simulation time: " << seconds << " seconds");
    std::cout << "\nTotal simulation time: " << seconds << " seconds\n";

    std::cout << "End of simulation, returning" << std::endl;

    return EXIT_SUCCESS;
}
