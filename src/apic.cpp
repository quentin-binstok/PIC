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
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ostream>

using json = nlohmann::json;
namespace fs = std::filesystem;



inline int particles_to_grid(
    particle_field *particles, float mp,
    scalar_field *vx, scalar_field *vy,
    scalar_field *mass_x, scalar_field *mass_y,
    std::ofstream &log_file)
{
    LOG_INFO(log_file, "APIC P->G (paper-faithful)");

    int nx = vx->nx, ny = vx->ny;
    float dx = vx->dx;

#pragma omp parallel for collapse(2)
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            vx->values[j*nx+i] = 0.0f;
            vy->values[j*nx+i] = 0.0f;
            mass_x->values[j*nx+i] = 0.0f;
            mass_y->values[j*nx+i] = 0.0f;
        }

#pragma omp parallel for
    for (int p = 0; p < particles->N; ++p) {

        float xp = particles->xyz[2*p];
        float yp = particles->xyz[2*p+1];

        float up = particles->velocity[2*p];
        float vp = particles->velocity[2*p+1];

        float bx0 = particles->bx[2*p];
        float bx1 = particles->bx[2*p+1];
        float by0 = particles->by[2*p];
        float by1 = particles->by[2*p+1];

        // --- X faces ---
        int i0 = (int)floor(xp/dx - 0.5f);
        int j0 = (int)floor(yp/dx);

        for (int j = j0; j <= j0+1; ++j)
        for (int i = i0; i <= i0+1; ++i) {

            if (i < 0 || j < 0 || i >= nx || j >= ny) continue;

            float xf = (i + 0.5f)*dx;
            float yf = j*dx;

            float w = kernel((xp-xf)/dx)*kernel((yp-yf)/dx);
            if (w == 0) continue;

            float ox = xf - xp;
            float oy = yf - yp;

            // Eq. (7): affine term
            float affine =
                (bx0 * particles->ix[3*p + 0] +
                 bx1 * particles->ix[3*p + 1]) * ox +
                (bx0 * particles->ix[3*p + 1] +
                 bx1 * particles->ix[3*p + 2]) * oy;

#pragma omp atomic
            vx->values[j*nx+i] += mp * w * (up + affine);
#pragma omp atomic
            mass_x->values[j*nx+i] += mp * w;
        }

        // --- Y faces ---
        i0 = (int)floor(xp/dx);
        j0 = (int)floor(yp/dx - 0.5f);

        for (int j = j0; j <= j0+1; ++j)
        for (int i = i0; i <= i0+1; ++i) {

            if (i < 0 || j < 0 || i >= nx || j >= ny) {
                continue;
            }
            float xf = i*dx;
            float yf = (j + 0.5f)*dx;

            float w = kernel((xp-xf)/dx)*kernel((yp-yf)/dx);
            if (w == 0) continue;

            float ox = xf - xp;
            float oy = yf - yp;

            float affine =
                (by0 * particles->iy[3*p + 0] +
                 by1 * particles->iy[3*p + 1]) * ox +
                (by0 * particles->iy[3*p + 1] +
                 by1 * particles->iy[3*p + 2]) * oy;

#pragma omp atomic
            vy->values[j*nx+i] += mp * w * (vp + affine);
#pragma omp atomic
            mass_y->values[j*nx+i] += mp * w;
        }
    }

#pragma omp parallel for collapse(2)
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            if (mass_x->values[j*nx+i] > 0)
                vx->values[j*nx+i] /= mass_x->values[j*nx+i];
            if (mass_y->values[j*nx+i] > 0)
                vy->values[j*nx+i] /= mass_y->values[j*nx+i];
        }

    return EXIT_SUCCESS;
}

inline int grid_to_particles(
    particle_field *particles,
    scalar_field *dom,
    scalar_field *vx, scalar_field *vy,
    Metrics &m,
    std::ofstream &log_file)
{
    LOG_INFO(log_file, "APIC G->P (paper-faithful)");

    int nx = vx->nx, ny = vx->ny;
    float dx = vx->dx;

#pragma omp parallel for
    for (int p = 0; p < particles->N; ++p) {

        float xp = particles->xyz[2*p];
        float yp = particles->xyz[2*p+1];

        float new_u = 0.0f, new_v = 0.0f;
        float bx0 = 0.0f, bx1 = 0.0f;
        float by0 = 0.0f, by1 = 0.0f;

        float Dx00=0, Dx01=0, Dx11=0;
        float Dy00=0, Dy01=0, Dy11=0;

        // --- X faces ---
        int i0 = (int)floor(xp/dx - 0.5f);
        int j0 = (int)floor(yp/dx);

        for (int j = j0; j <= j0+1; ++j)
        for (int i = i0; i <= i0+1; ++i) {

            if (i < 0 || j < 0 || i >= nx || j >= ny) continue;

            float xf = (i + 0.5f)*dx;
            float yf = j*dx;
            float w = kernel((xp-xf)/dx)*kernel((yp-yf)/dx);
            if (w == 0) continue;

            float ox = xf - xp;
            float oy = yf - yp;

            float vi = vx->values[j*nx+i];

            new_u += w * vi;
            bx0 += w * vi * ox;
            bx1 += w * vi * oy;

            Dx00 += w * ox*ox;
            Dx01 += w * ox*oy;
            Dx11 += w * oy*oy;
        }

        // --- Y faces ---
        i0 = (int)floor(xp/dx);
        j0 = (int)floor(yp/dx - 0.5f);

        for (int j = j0; j <= j0+1; ++j)
        for (int i = i0; i <= i0+1; ++i) {

            if (i < 0 || j < 0 || i >= nx || j >= ny) continue;

            float xf = i*dx;
            float yf = (j + 0.5f)*dx;
            float w = kernel((xp-xf)/dx)*kernel((yp-yf)/dx);
            if (w == 0) continue;

            float ox = xf - xp;
            float oy = yf - yp;

            float vi = vy->values[j*nx+i];

            new_v += w * vi;
            by0 += w * vi * ox;
            by1 += w * vi * oy;

            Dy00 += w * ox*ox;
            Dy01 += w * ox*oy;
            Dy11 += w * oy*oy;
        }

        particles->velocity[2*p]   = new_u;
        particles->velocity[2*p+1] = new_v;
        particles->bx[2*p]   = bx0;
        particles->bx[2*p+1] = bx1;
        particles->by[2*p]   = by0;
        particles->by[2*p+1] = by1;

        int i = std::max(0, std::min(nx - 1, (int)(xp / dx)));
        int j = std::max(0, std::min(ny - 1, (int)(yp / dx)));

        float cell_type = GET(dom, i, j);
        float left_cell = GET(dom, i - 1, j);
        float right_cell = GET(dom, i + 1, j) ;
        float bottom_cell = GET(dom, i, j - 1);
        float top_cell = GET(dom, i, j + 1);

        // Invert D (Eq. 6)
        float detDx = Dx00*Dx11 - Dx01*Dx01;
        float detDy = Dy00*Dy11 - Dy01*Dy01;

        if (detDx < 1e-15f) {
            particles->ix[3*p+0] = 0.0f; // arbitrary large value to make affine term negligible
            particles->ix[3*p+1] = 0.0f;
            particles->ix[3*p+2] = 0.0f;
            m.singularity_count++;
        } else if (cell_type == DIRICHLET || left_cell == DIRICHLET || right_cell == DIRICHLET || bottom_cell == DIRICHLET || top_cell == DIRICHLET){ // avoid singularity in non-liquid cells (treat as PIC)
            particles->ix[3*p] =  4.0f/(dx*dx); // arbitrary large value to make affine term negligible
            particles->ix[3*p + 1] =  0.0f;
            particles->ix[3*p + 2] =  4.0f/(dx*dx);
            m.dirichlet++;
        } else {
            particles->ix[3*p+0] =  Dx11/detDx;
            particles->ix[3*p+1] = -Dx01/detDx;
            particles->ix[3*p+2] =  Dx00/detDx;
        }

        if (detDy < 1e-15f) {
            particles->iy[3*p+0] = 0.0f; // arbitrary large value to make affine term negligible
            particles->iy[3*p+1] = 0.0f;
            particles->iy[3*p+2] = 0.0f;
            m.singularity_count++;
        } else if (cell_type == DIRICHLET || left_cell == DIRICHLET || right_cell == DIRICHLET || bottom_cell == DIRICHLET || top_cell == DIRICHLET){ // avoid singularity in non-liquid cells (treat as PIC)
            particles->iy[3*p] =  4.0f/(dx*dx); // arbitrary large value to make affine term negligible
            particles->iy[3*p + 1] =  0.0f;
            particles->iy[3*p + 2] =  4.0f/(dx*dx);
            m.dirichlet++;
        } else {
            particles->iy[3*p+0] =  Dy11/detDy;
            particles->iy[3*p+1] = -Dy01/detDy;
            particles->iy[3*p+2] =  Dy00/detDy;
        }
    }

    return EXIT_SUCCESS;
}

/*
 @brief the APIC solver
 @param data: the whole json
 @param log_file: the log file
 @param metrics_file: the metrics file
*/
int solver_apic(json &data, std::ofstream &log_file,
                std::ofstream &metrics_file, fs::path work_dir) {
    LOG_INFO(log_file, "Starting the APIC solver");

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
    bool gravity = data.value("gravity", false);
    float g = data.value("g", 9.81);
    float init_temp = data.value("init_temperature", 20);
    float T0 = data.value("T0", 20);
    float beta = data.value("beta", 0.01);
    float c = data.value("c", 4.186);
    float k = data.value("k", 1.0f);
    RNG rng(dx, dt);
    float mass = rho * (dx * dx) / particle_density;
    bool thermal = data.value("thermal", false);

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
    scalar_field *kern_sum_T =
        scalar_field_init("kern_sum_T", nx, ny, 0, 0, dx, log_file);

    scalar_field *mass_x =
        scalar_field_init("mass_x", nx, ny, 0, 0, dx, log_file);
    scalar_field *mass_y =
        scalar_field_init("mass_y", nx, ny, 0, 0, dx, log_file);

    if (!vx || !vy || !p || !div || !dom || !temp_vx || !temp_vy || !temp_p ||
        !mass_x || !mass_y) {
        LOG_ERR(log_file, "An error occured initializing fields.");
        return EXIT_FAILURE;
    }

    std::vector<float> speed_condition;
    therm_bc *therm_bcs = (therm_bc *)malloc(sizeof(therm_bc));

    // Applying the initial conditions
    initialize_speed(vx, dom, data, "ic_vx", log_file);
    initialize_speed(vy, dom, data, "ic_vy", log_file);
    boundary_condition(vx, vy, dom, speed_condition, data, "bc", log_file);
    initialize_domain(dom, data, "ic_cell", log_file);
    create_circle(dom, "ic_cylinders", data, log_file);
    initialize_taylor_green_vortex(vx, vy, dom, data, "taylor_green", log_file);
    /* build_thermal_bc(therm_bcs, data, log_file); */

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

    auto t7 = std::chrono::high_resolution_clock::now();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto t3 = std::chrono::high_resolution_clock::now();
    auto t4 = std::chrono::high_resolution_clock::now();
    auto t5 = std::chrono::high_resolution_clock::now();
    auto t6 = std::chrono::high_resolution_clock::now();

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

        t7 = std::chrono::high_resolution_clock::now();

        std::fill(density.begin(), density.end(), 0);
        check_particles(particles, dom, m, density, log_file);
        refill_domain(particles, dom, vx, vy, T, density, particle_density,
                      refill, creation_rate, dt, rng, m, log_file);

        t2 = std::chrono::high_resolution_clock::now();

        if (gravity)
            apply_gravity(particles, g, dt, beta, T0);

        particles_to_grid(particles, mass, vx, vy, mass_x, mass_y, log_file);
        particles_temp_to_grid(particles, T, kern_sum_T, log_file);
        
        t3 = std::chrono::high_resolution_clock::now();
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

        if (thermal)
            apply_thermal_eq(T, T_temp, therm_bcs, dt, c, rho, k, tol, max_iter,
                             log_file);

        project_velocity(p, vx, vy, dom, dx, dt, rho, log_file,
                         speed_condition);

        t4 = std::chrono::high_resolution_clock::now();

        // This is to be able to save it. It serves no purpose in the
        // algorithm
        divergence(vx, vy, div, dom, speed_condition, log_file);

        grid_to_particles(particles, dom, vx, vy, m, log_file);


        t5 = std::chrono::high_resolution_clock::now();

        advect(particles, vx, vy, dt, log_file);
        
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

        singularity = m.singularity_count;
        solid_particles = m.particle_in_solid;
        dirichlet = m.dirichlet;

        m = compute_metrics(dom, p, vx, vy, div, dx, i, nt, singularity, solid_particles, dirichlet, m, data["metrics"], log_file);
        write_metrics(metrics_file, m);

        first_loop = false;

        t6 = std::chrono::high_resolution_clock::now();
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

    scalar_field_free(mass_x, log_file);
    scalar_field_free(mass_y, log_file);
    scalar_field_free(kern_sum_T, log_file);

    delete particles;
    free(therm_bcs);

    auto t1 = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(t1 - t0).count();
    LOG_INFO(log_file, "Total simulation time: " << seconds << " seconds");
    std::cout << "\nTotal simulation time: " << seconds << " seconds\n";

    std::cout << "End of simulation, returning" << std::endl;

    return EXIT_SUCCESS;
}