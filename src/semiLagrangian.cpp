#include "conditions.hpp"
#include "data.hpp"
#include "nlohmann/json.hpp"
#include "poisson.hpp"
#include "utils.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

using json = nlohmann::json;
namespace fs = std::filesystem;

/*
 @brief Applies semi-lagrangian advection to the field q
 @param vx, vy: the velocity field
 @param dt: the time step
 @param q: the scalar field to advect.
 @param log_file: the log file
*/
inline int advect(scalar_field *vx, scalar_field *vy, float dt,
                  scalar_field *q_n, scalar_field *q_n1,
                  std::ofstream &log_file) {
    LOG_INFO(log_file, "Advecting field " << q_n->name);
    unsigned int nx = q_n->nx, ny = q_n->ny;
    float x_int = q_n->x_internal, y_int = q_n->y_internal;
    float dx = q_n->dx;
#pragma omp parallel for collapse(2)
    for (unsigned int j = 0; j < ny; j++) {
        for (unsigned int i = 0; i < nx; i++) {
            // Interpolation of the speed field
            // coords where we need the speed field
            float x = (i + x_int) * dx, y = (j + y_int) * dx;
            // vx
            float v_x = 0;
            int x_1, x_2, y_1, y_2;
            x_1 = (int)(x / dx - 0.5);
            x_1 = std::max(0, x_1);
            if (x_1 > vx->nx - 1)
                LOG_WARN(log_file, "x_1 too big")
            x_2 = x_1 + 1;
            x_2 = std::min(vx->nx - 1, x_2);
            y_1 = (int)(y / dx);
            y_1 = std::max(0, y_1);
            y_1 = std::min(y_1, vx->ny - 2);
            if (y_1 > vx->ny - 1)
                LOG_WARN(log_file,
                         "y_1 too big, y = " << y << " and y_1 = " << y_1)
            y_2 = y_1 + 1;
            y_2 = std::min(vx->ny - 1, y_2);
            v_x = interpolate_bilinear(
                x, y, (x_1 + vx->x_internal) * dx, (y_1 + vx->y_internal) * dx,
                GET(vx, x_1, y_1), GET(vx, x_2, y_1), GET(vx, x_1, y_2),
                GET(vx, x_2, y_2), dx, dx);
            // vy
            float v_y = 0;
            x_1 = (int)(x / dx);
            x_1 = std::max(0, x_1);
            x_1 = std::min(x_1, vy->nx - 2);
            if (x_1 > vy->nx - 1)
                LOG_WARN(log_file, "x_1 too big")
            x_2 = x_1 + 1;
            x_2 = std::min(vy->nx - 1, x_2);
            y_1 = (int)(y / dx - 0.5);
            y_1 = std::max(0, y_1);
            if (y_1 > vy->ny - 1)
                LOG_WARN(log_file, "y_1 too big")
            y_2 = y_1 + 1;
            y_2 = std::min(vy->ny - 1, y_2);
            v_y = interpolate_bilinear(
                x, y, (x_1 + vy->x_internal) * dx, (y_1 + vy->y_internal) * dx,
                GET(vy, x_1, y_1), GET(vy, x_2, y_1), GET(vy, x_1, y_2),
                GET(vy, x_2, y_2), dx, dx);
            // xp
            float xp_x = x - dt * v_x;
            float xp_y = y - dt * v_y;
            xp_x = std::max((float)0, xp_x);
            xp_y = std::max((float)0, xp_y);
            xp_x = std::min((q_n->nx - 1) * dx, xp_x);
            xp_y = std::min((q_n->ny - 1) * dx, xp_y);
            x_1 = (int)(xp_x / dx - q_n->x_internal);
            x_1 = std::max(0, x_1);
            x_2 = x_1 + 1;
            x_2 = std::min(q_n->nx - 1, x_2);
            y_1 = (int)(xp_y / dx - q_n->y_internal);
            y_1 = std::max(0, y_1);
            y_2 = y_1 + 1;
            y_2 = std::min(q_n->ny - 1, y_2);
            // Get q at xp
            // vx
            float q_interp = 0;
            q_interp = interpolate_bilinear(
                xp_x, xp_y, (x_1 + x_int) * dx, (y_1 + y_int) * dx,
                GET(q_n, x_1, y_1), GET(q_n, x_2, y_1), GET(q_n, x_1, y_2),
                GET(q_n, x_2, y_2), dx, dx);
            SET(q_n1, i, j, q_interp);
        }
    }
    return EXIT_SUCCESS;
}

void apply_gravity_SL(scalar_field *vy, scalar_field *dom, float dt, float g,
                      std::ofstream &log_file) {
    LOG_INFO(log_file, "Applying gravity");
    int nx = vy->nx, ny = vy->ny;

#pragma omp parallel for collapse(2)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            CELL_TYPE cell = (CELL_TYPE)GET(dom, i, j);

            CELL_TYPE top_cell = LIQUID;
            if (j != ny - 1)
                top_cell = (CELL_TYPE)GET(dom, i, j + 1);

            if (cell == LIQUID && (top_cell == LIQUID || top_cell == AIR))
                SET(vy, i, j, GET(vy, i, j) - dt * g);
        }
    }
}

/*
 @brief the semi lagrangian solver
 @param data: the whole json
 @param log_file: the log file
 @param metrics_file: the metrics file
 @param work_dir: the working directory
*/
int solver_semi_lagrangian(json &data, std::ofstream &log_file, std::ofstream &metrics_file,
                           fs::path work_dir) {
    LOG_INFO(log_file, "Starting the semi-lagrangian solver");
    auto t0 = std::chrono::high_resolution_clock::now();
    if (check_params(data, log_file)) {
        LOG_ERR(log_file, "Problem checking the input parameters")
        return EXIT_FAILURE;
    }

    // Getting based params
    unsigned int nx = data["grid"][0], ny = data["grid"][1];
    float dx = data["space_steps"];
    int sampling_rate = data["sampling_rate"];
    float dt = data.value("delta_t", 0.1);
    unsigned int nt = data.value("nt", 10);
    float rho = data.value("rho", 1000);
    float tol = data.value("tol", 1e-5);
    int max_iter = data.value("max_iter", 1e5);
    bool gravity = data.value("gravity", false);
    float g = data.value("g", 9.81);

    std::vector<float> speed_condition;

    // Initialising the fields
    scalar_field *vx = scalar_field_init("vx", nx, ny, 0.5, 0, dx, log_file);
    scalar_field *vy = scalar_field_init("vy", nx, ny, 0, 0.5, dx, log_file);
    scalar_field *p = scalar_field_init("p", nx, ny, 0, 0, dx, log_file);
    scalar_field *div = scalar_field_init("div", nx, ny, 0, 0, dx, log_file);
    scalar_field *dom = scalar_field_init("dom", nx, ny, 0, 0, dx, log_file);
    scalar_field *temp_vx = scalar_field_copy(vx, log_file);
    scalar_field *temp_vy = scalar_field_copy(vy, log_file);
    scalar_field *temp_p = scalar_field_copy(p, log_file);

    if (!vx || !vy || !p || !div || !dom || !temp_vy || !temp_vy || !temp_p) {
        LOG_ERR(log_file, "An error occured initializing fields.");
        return EXIT_FAILURE;
    }

    // Applying the initial conditions
    initialize_domain(dom, data, "ic_cell", log_file);
    initialize_speed(vx, dom, data, "ic_vx", log_file);
    initialize_speed(vy, dom, data, "ic_vy", log_file);
    boundary_condition(vx, vy, dom, speed_condition, data, "bc", log_file);
    initialize_taylor_green_vortex(vx, vy, dom, data, "taylor_green", log_file);
    create_circle(dom, "ic_cylinders", data, log_file);

    // Manifests
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

    // Initial time step
    write_scalar_vtk(vx, 0, 0, log_file, work_dir);
    write_scalar_vtk(vy, 0, 0, log_file, work_dir);
    write_scalar_vtk(p, 0, 0, log_file, work_dir);
    write_scalar_vtk(div, 0, 0, log_file, work_dir);
    write_scalar_vtk(dom, 0, 0, log_file, work_dir);

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

    // Main time loop
    bool inverted = false;
    bool first_loop = true;
    std::cout << "Starting simulation" << std::endl;
    for (unsigned int i = 1; i < nt; i++) {
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
            apply_gravity_SL(vy, dom, dt, g, log_file);

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

        project_velocity(p, vx, vy, dom, dx, dt, rho, log_file,
                         speed_condition);

        // This is to be able to save it. It serves no purpose in the
        // algorithm
        divergence(vx, vy, div, dom, speed_condition, log_file);
        // save files
        if (sampling_rate && !(i % sampling_rate)) {
            write_scalar_vtk(vx, i, 0, log_file, work_dir);
            write_scalar_vtk(vy, i, 0, log_file, work_dir);
            write_scalar_vtk(p, i, 0, log_file, work_dir);
            write_scalar_vtk(div, i, 0, log_file, work_dir);
            write_scalar_vtk(dom, i, 0, log_file, work_dir);
        }

        // advect
        advect(vx, vy, dt, vx, temp_vx, log_file);
        advect(vx, vy, dt, vy, temp_vy, log_file);

        m = compute_metrics(dom, p, vx, vy, div, dx, i, nt, singularity, solid_particles, dirichlet, m, data["metrics"], log_file);
        write_metrics(metrics_file, m);

        inverted = !inverted;
        scalar_field *invert_vx = vx;
        scalar_field *invert_vy = vy;
        vx = temp_vx;
        vy = temp_vy;
        temp_vx = invert_vx;
        temp_vy = invert_vy;
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

    auto t1 = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(t1 - t0).count();
    LOG_INFO(log_file, "Total simulation time: " << seconds << " seconds");
    std::cout << "\nTotal simulation time: " << seconds << " seconds\n";

    std::cout << "End of simulation, returning" << std::endl;
    return EXIT_SUCCESS;
}