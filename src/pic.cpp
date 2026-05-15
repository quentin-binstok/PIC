#include "conditions.hpp"
#include "data.hpp"
#include "nlohmann/json.hpp"
#include "particules.hpp"
#include "poisson.hpp"
#include "thermal.hpp"
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

/*
 @brief the PIC/FLIP
 @param data: the whole json
 @param log_file: the log file
 @param metrics_file: the metrics file
*/
int solver_pic(json &data, std::ofstream &log_file, std::ofstream &metrics_file,
               fs::path work_dir) {
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
    float tol = data.value("tol", 1e-5);
    float tol_therm = data.value("tol_therm", tol);
    int max_iter = data.value("max_iter", 1e5);
    int particle_density = data.value("particle_density", 8);
    bool refill = data.value("refill", false);
    float flip_param = data.value("flip", 0.0f);
    LOG_INFO(log_file, "FLIP percentage is " << flip_param * 100);
    bool gravity = data.value("gravity", false);
    float g = data.value("g", 9.81);
    float init_temp = data.value("init_temperature", 20);
    float T0 = data.value("T0", 20);
    float beta = data.value("beta", 0.01);

    bool thermal = data.value("thermal", false);
    float c_liq = data.value("c", 4.186);
    float k_liq = data.value("k", 1.0f);
    float rho_liq = data.value("rho", 1000.0f);
    float c_air = data.value("c_air", c_liq);
    float k_air = data.value("k_air", k_liq);
    float rho_air = data.value("rho_air", rho_liq);
    float c_sol = data.value("c_sol", c_liq);
    float k_sol = data.value("k_sol", k_liq);
    float rho_sol = data.value("rho_sol", rho_liq);

    uint32_t seed = data.value("seed", std::random_device{}());
    LOG_INFO(log_file, "Seed is " << seed);

    RNG rng(dx, dt, seed);

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

    scalar_field *T =
        scalar_field_init("Temperature", nx, ny, 0, 0, dx, log_file);
    scalar_field *T_temp = scalar_field_copy(T, log_file);
    scalar_field *r =
        scalar_field_init("termal_gen", nx, ny, 0, 0, dx, log_file);

    scalar_field *kern_sum_vx =
        scalar_field_init("kern_sum_vx", nx, ny, 0, 0, dx, log_file);
    scalar_field *kern_sum_vy =
        scalar_field_init("kern_sum_vy", nx, ny, 0, 0, dx, log_file);
    scalar_field *kern_sum_T =
        scalar_field_init("kern_sum_T", nx, ny, 0, 0, dx, log_file);

    if (!vx || !vy || !p || !div || !dom || !temp_vx || !temp_vy || !temp_p ||
        !kern_sum_vx || !kern_sum_vy || !r) {
        LOG_ERR(log_file, "An error occured initializing fields.");
        return EXIT_FAILURE;
    }

    if (thermal)
        initialize_thermal_generation(r, data, log_file);

    std::vector<float> speed_condition;
    therm_bc *therm_bcs = new therm_bc;

    // Applying the initial conditions
    initialize_speed(vx, dom, data, "ic_vx", log_file);
    initialize_speed(vy, dom, data, "ic_vy", log_file);
    boundary_condition(vx, vy, dom, speed_condition, data, "bc", log_file);
    initialize_domain(dom, data, "ic_cell", log_file);
    create_circle(dom, "ic_cylinders", data, log_file);
    initialize_taylor_green_vortex(vx, vy, dom, data, "taylor_green", log_file);

    if (thermal)
        build_thermal_bc(therm_bcs, data, log_file);

    if (data.contains("special")) {
        if (data["special"]["type"] == "sine")
            sine_surface(data, dom, log_file);
        if (data["special"]["type"] == "taylor green")
            initialize_taylor_green_vortex(vx, vy, dom, data, "taylor_green",
                                           log_file);
    }

#pragma omp parallel for collapse(2)
    for (int j = 0; j < (int)ny; j++)
        for (int i = 0; i < (int)nx; i++)
            SET(T, i, j, init_temp);

    // Initializing the particles
    particle_field *particles = particle_field_init_2D(
        "particles", nx * ny * particle_density, log_file);
    initialize_particles(particles, dom, vx, vy, T, data["particle_density"],
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
    if (thermal)
        write_manifest_vtk(T->name, dt, nt, sampling_rate, 1, 0, log_file,
                           work_dir);

    // Initial state
    write_scalar_vtk(vx, 0, 0, log_file, work_dir);
    write_scalar_vtk(vy, 0, 0, log_file, work_dir);
    write_scalar_vtk(p, 0, 0, log_file, work_dir);
    write_scalar_vtk(div, 0, 0, log_file, work_dir);
    write_scalar_vtk(dom, 0, 0, log_file, work_dir);
    if (thermal)
        write_scalar_vtk(T, 0, 0, log_file, work_dir);

    std::vector<int> density(nx * ny, 0);

    // Main time loop
    // bool inverted = false;
    bool first_loop = true;
    std::cout << "Starting simulation" << std::endl;

    Metrics m;
    std::vector<std::string> headers = build_headers(data["metrics"]);
    m.singularity_count = 0;
    m.particle_in_solid = 0;
    m.dirichlet = 0;
    int singularity = 0;
    int solid_particles = 0;
    int dirichlet = 0;
    m = compute_metrics(dom, p, vx, vy, div, dx, 0, nt, singularity, solid_particles, dirichlet, m, data["metrics"], log_file);
    write_header(metrics_file, headers);
    write_metrics(metrics_file, m);
    
    
    auto slice_files = init_slice_csvs(data, work_dir);

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
            std::cout << "Iteration " << i << "/" << nt << "\n";
            std::flush(std::cout);
        }

        m.singularity_count = 0;
        m.particle_in_solid = 0;
        m.dirichlet = 0;

        if (gravity)
            apply_gravity(particles, g, dt, beta, T0);

        particles_speed_to_grid(particles, vx, vy, kern_sum_vx, kern_sum_vy,
                                log_file);
        if (thermal)
            particles_temp_to_grid(particles, T, kern_sum_T, log_file);

        divergence(vx, vy, div, dom, speed_condition, log_file);

        if (data["iteration_algo"] == "Jacobi")
            jacobi(p, temp_p, div, vx, vy, dom, tol, dt, rho_liq, max_iter,
                   first_loop, log_file);
        else if (data["iteration_algo"] == "SOR")
            sor(p, div, vx, vy, dom, tol, dt, rho_liq, max_iter, log_file);
        else {
            LOG_ERR(log_file, "Iteration algorithm not supported");
            return EXIT_FAILURE;
        }

        if (thermal)
            apply_thermal_eq(T, T_temp, r, dom, therm_bcs, dt, c_liq, c_air,
                             c_sol, rho_liq, rho_air, rho_sol, k_liq, k_air,
                             k_sol, tol_therm, max_iter, log_file);

        // Needed for FLIP
        std::memcpy(temp_vx->values, vx->values, nx * ny * sizeof(float));
        std::memcpy(temp_vy->values, vy->values, nx * ny * sizeof(float));

        project_velocity(p, vx, vy, dom, dx, dt, rho_liq, log_file,
                         speed_condition);

        // This is to be able to save it. It serves no purpose in the
        // algorithm
        divergence(vx, vy, div, dom, speed_condition, log_file);

        grid_speed_to_particles(particles, vx, vy, temp_vx, temp_vy, flip_param,
                                log_file);
        if (thermal)
            grid_temp_to_particles(particles, T, log_file);

        // save files
        if (sampling_rate && !(i % sampling_rate)) {
            write_scalar_vtk(vx, i, 0, log_file, work_dir);
            write_scalar_vtk(vy, i, 0, log_file, work_dir);
            write_scalar_vtk(p, i, 0, log_file, work_dir);
            write_scalar_vtk(div, i, 0, log_file, work_dir);
            write_scalar_vtk(dom, i, 0, log_file, work_dir);
            if (thermal)
                write_scalar_vtk(T, i, 0, log_file, work_dir);

            write_particles_vtp(particles, i, 0, 2, log_file, work_dir);
        }

        advect(particles, vx, vy, dt, log_file);

        std::fill(density.begin(), density.end(), 0);
        check_particles(particles, dom, m, density, log_file);
        refill_domain(particles, dom, vx, vy, T, density, particle_density,
                      refill, creation_rate, dt, rng, m, log_file);

        m = compute_metrics(dom, p, vx, vy, div, dx, i, nt, singularity, solid_particles, dirichlet, m, data["metrics"], log_file);
        write_metrics(metrics_file, m);

        
        
        for (const auto& [key, cfg] : data.items()) {
                if (key.rfind("slice_", 0) != 0)
                    continue;


                if (!should_write_slice_csv(i, cfg))
                    continue;

                const std::string field = cfg["field"].get<std::string>();
                std::ofstream& f = slice_files[key];

                scalar_field* data = nullptr;
                if (field == "vx") data = vx;
                else if (field == "vy") data = vy;

                if (!data)
                    continue;

                write_slice_vertical_csv(
                    f,
                    i,
                    data,
                    cfg["i"].get<int>(),
                    cfg["j_start"].get<int>(),
                    cfg["j_end"].get<int>()
                );
            }

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
    scalar_field_free(T, log_file);
    scalar_field_free(T_temp, log_file);

    scalar_field_free(kern_sum_vx, log_file);
    scalar_field_free(kern_sum_vy, log_file);
    scalar_field_free(kern_sum_T, log_file);

    free(particles);
    delete therm_bcs;

    auto t1 = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(t1 - t0).count();
    LOG_INFO(log_file, "Total simulation time: " << seconds << " seconds");
    std::cout << "\nTotal simulation time: " << seconds << " seconds\n";

    std::cout << "End of simulation, returning" << std::endl;

    return EXIT_SUCCESS;
}
