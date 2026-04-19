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

inline int particles_to_grid(particle_field *particles, float mass,
                             scalar_field *vx, scalar_field *vy,
                             scalar_field *mass_x,
                             scalar_field *mass_y, // NEW: grid mass per face
                             std::ofstream &log_file) {
    LOG_INFO(log_file, "APIC P->G transfer");

    int nx = vx->nx, ny = vx->ny;
    float dx = vx->dx;
    float mp = mass;

    // Zero all grid fields
#pragma omp parallel for collapse(2)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            SET(vx, i, j, 0);
            SET(vy, i, j, 0);
            SET(mass_x, i, j, 0);
            SET(mass_y, i, j, 0);
        }
    }

#pragma omp parallel for
    for (int k = 0; k < particles->N; k++) {
        float x = particles->xyz[2 * k];
        float y = particles->xyz[2 * k + 1];

        float u = particles->velocity[2 * k];
        float v = particles->velocity[2 * k + 1];

        // computed previous time step
        float C00 = particles->C[4 * k];
        float C01 = particles->C[4 * k + 1];
        float C10 = particles->C[4 * k + 2];
        float C11 = particles->C[4 * k + 3];

        // --- x-faces: staggered in x, collocated in y ---
        int i_vx = std::max(0, std::min(nx - 1, (int)(x / dx - 0.5f)));
        int j_vx = std::max(0, std::min(ny - 1, (int)(y / dx)));
        for (int j = j_vx; j <= j_vx + 1; j++) {
            for (int i = i_vx; i <= i_vx + 1; i++) {

                if (i < 0 || i >= nx || j < 0 || j >= ny) continue;

                float kern = kernel((x - (i + 0.5f) * dx) / dx) *
                             kernel((y -  j          * dx) / dx);
                
                float ox = (i + 0.5f) * dx - x;
                float oy = j * dx - y;

                // Affine part 
                float affine = C00 * ox + C01 * oy;


#pragma omp atomic
                vx->values[j * nx + i] += mp * kern * (u + affine);

                // Eq 5
#pragma omp atomic
                mass_x->values[j * nx + i] += mp * kern;
            }
        }

        // --- y-faces: collocated in x, staggered in y ---
        int i_vy = std::max(0, std::min(nx - 1, (int)(x / dx )));
        int j_vy = std::max(0, std::min(ny - 1, (int)(y / dx -0.5f)));
        for (int j = j_vy; j <= j_vy + 1; j++) {
            for (int i = i_vy; i <= i_vy + 1; i++) {

                if (i < 0 || i >= nx || j < 0 || j >= ny) continue;

                float kern = kernel((x - i * dx) / dx) *
                             kernel((y - (j + 0.5f) * dx) / dx);

                float ox = i * dx - x;
                float oy = (j + 0.5f) * dx - y;

                float affine = C10 * ox + C11 * oy;
#pragma omp atomic
                vy->values[j * nx + i] += mp * kern * (v + affine);

#pragma omp atomic
                mass_y->values[j * nx + i] += mp * kern;
            }
        }
    }

    // Divide momentum by mass to get velocity
#pragma omp parallel for collapse(2)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            float mx = GET(mass_x, i, j);
            float my = GET(mass_y, i, j);
            if (mx > 0)
                SET(vx, i, j, GET(vx, i, j) / mx);
            if (my > 0)
                SET(vy, i, j, GET(vy, i, j) / my);
        }
    }

    return EXIT_SUCCESS;
}


inline int grid_to_particles(
    particle_field *particles,
    scalar_field *vx, scalar_field *vy, Metrics &m, std::ofstream &log_file)
{
    LOG_INFO(log_file, "APIC G->P transfer");

    int nx = vx->nx, ny = vx->ny;
    float dx = vx->dx;

#pragma omp parallel for
    for (int k = 0; k < particles->N; k++) {
        float x = particles->xyz[2 * k];
        float y = particles->xyz[2 * k + 1];

        float new_u = 0, new_v = 0;

        // bx = row vector, Dx = 2x2 matrix (x-faces only)
        float bx_0 = 0, bx_1 = 0;
        float by_0 = 0, by_1 = 0;


        float Dx_00 = 0, Dx_01 = 0, Dx_11 = 0;  // symmetric: D10 = D01
        float Dy_00 = 0, Dy_01 = 0, Dy_11 = 0;  // symmetric: D10 = D01

        int i_vx = std::max(0, (int)(x / dx - 0.5)); // matches get_speed
        int j_vx = std::max(0, (int)(y / dx));
        for (int j = j_vx; j < std::min(ny, j_vx + 2); j++) {
            for (int i = i_vx; i < std::min(nx, i_vx + 2); i++) {
                
                float kern = kernel((x - (i + 0.5f) * dx) / dx) *
                             kernel((y -  j          * dx) / dx);
                float vi = GET(vx, i, j);

                float ox = (i + 0.5f) * dx - x;
                float oy = j * dx - y;

                new_u += kern * vi;

                // bx += w * vx * o  (vector)
                bx_0 += kern * vi * ox;
                bx_1 += kern * vi * oy;

                // D00 += w * o * o^T  (symmetric matrix)
                Dx_00 += kern * ox * ox;
                Dx_01 += kern * ox * oy;
                Dx_11 += kern * oy * oy;
            }
        }

        // --- y-faces ---
        int i_vy = std::max(0, (int)(x / dx));
        int j_vy = std::max(0, (int)(y / dx - 0.5f));
        for (int j = j_vy; j < std::min(ny, j_vy + 2); j++) {
            for (int i = i_vy; i < std::min(nx, i_vy + 2); i++) {

                float kern = kernel((x - i * dx) / dx) *
                             kernel((y - (j + 0.5f) * dx) / dx);
                float vi = GET(vy, i, j);

                float ox = i * dx - x;
                float oy = (j + 0.5f) * dx - y;

                new_v += kern * vi;

                // by_0 += w * vy * o  (vector)
                by_0 += kern * vi * ox;
                by_1 += kern * vi * oy;

                Dy_00 += kern * ox * ox;
                Dy_01 += kern * ox * oy;
                Dy_11 += kern * oy * oy;
            }
        }

        particles->velocity[2 * k] = new_u;
        particles->velocity[2 * k + 1] = new_v;

        // Invert D (symmetric 2x2)
        float det_Dx = Dx_00 * Dx_11 - Dx_01 * Dx_01;

        float det_Dy = Dy_00 * Dy_11 - Dy_01 * Dy_01;

        if (det_Dx > 1e-15 && i_vx < nx - 1 && j_vx < ny - 1 && j_vx >= 0 && i_vx >= 0){ // avoid singularity (treat as PIC)
            // D^-1
            float ix_00 =  Dx_11 / det_Dx;
            float ix_01 = -Dx_01 / det_Dx;
            float ix_11 =  Dx_00 / det_Dx;

            // Cx = bx^T * Dx^-1  (row 0 of C)
            particles->C[4*k]     = bx_0 * ix_00 + bx_1 * ix_01;  // C00
            particles->C[4*k + 1] = bx_0 * ix_01 + bx_1 * ix_11;  // C01
        } else {
            particles->C[4*k]     = 0.0f;
            particles->C[4*k + 1] = 0.0f;
            m.singularity_count++;
        }
        if (det_Dy > 1e-15 && i_vy < nx - 1 && j_vy < ny - 1 && j_vy >= 0 && i_vy >= 0){ // avoid singularity (treat as PIC)
            float iy_00 =  Dy_11 / det_Dy;
            float iy_01 = -Dy_01 / det_Dy;
            float iy_11 =  Dy_00 / det_Dy;

            particles->C[4*k + 2] = by_0 * iy_00 + by_1 * iy_01;  // C10
            particles->C[4*k + 3] = by_0 * iy_01 + by_1 * iy_11;  // C11
        } else {
            particles->C[4*k + 2] = 0.0f;
            particles->C[4*k + 3] = 0.0f;
            m.singularity_count++;
        }

    }
    return EXIT_SUCCESS;
}


/* inline int particles_to_grid(
    particle_field *particles, float mass,
    scalar_field *vx, scalar_field *vy,
    scalar_field *mass_x, scalar_field *mass_y,  // NEW: grid mass per face
    std::ofstream &log_file)
{
    LOG_INFO(log_file, "APIC P->G transfer");

    int nx = vx->nx, ny = vx->ny;
    float dx = vx->dx;
    float mp = mass;

    // Zero all grid fields
#pragma omp parallel for collapse(2)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            SET(vx,      i, j, 0);
            SET(vy,      i, j, 0);
            SET(mass_x, i, j, 0);  
            SET(mass_y, i, j, 0);  
        }
    }

#pragma omp parallel for
    for (int k = 0; k < particles->N; k++) {
        float x  = particles->xyz[2 * k];
        float y  = particles->xyz[2 * k + 1];

        float u = particles->velocity[2 * k];    
        float v = particles->velocity[2 * k + 1];  

        // computed previous time step
        float B00 = particles->B[4 * k];
        float B01 = particles->B[4 * k + 1];
        float B10 = particles->B[4 * k + 2];
        float B11 = particles->B[4 * k + 3];

        // --- x-faces: staggered in x, collocated in y ---
        int i_vx = std::max(0, (int)(x / dx - 0.5)); // matches get_speed
        int j_vx = std::max(0, (int)(y / dx));
        for (int j = j_vx; j < std::min(ny, j_vx + 2); j++) {
            for (int i = i_vx; i < std::min(nx, i_vx + 2); i++) {

                float kern = kernel((x - (i + 0.5f) * dx) / dx) *
                             kernel((y -  j          * dx) / dx);

                float grad_x = kernel_grad((x - (i + 0.5f) * dx) / dx) *
                               kernel((y -  j          * dx) / dx) / dx;
                float grad_y = kernel((x - (i + 0.5f) * dx) / dx) *
                                kernel_grad((y -  j          * dx) / dx) / dx;

                // Affine part 
                float affine = B00 * grad_x + B01 * grad_y;

                // velocity part
                float velocity = kern * u;


#pragma omp atomic
                vx->values[j * nx + i] += mp * (velocity + affine);

                // Eq 5
#pragma omp atomic
                mass_x->values[j * nx + i] += mp * kern;
            }
        }

        // --- y-faces: collocated in x, staggered in y ---
        int i_vy = std::max(0, (int)(x / dx));
        int j_vy = std::max(0, (int)(y / dx - 0.5f));
        for (int j = j_vy; j < std::min(ny, j_vy + 2); j++) {
            for (int i = i_vy; i < std::min(nx, i_vy + 2); i++) {
                float kern = kernel((x -  i         * dx) / dx) *
                             kernel((y - (j + 0.5f) * dx) / dx);
                float grad_x = kernel_grad((x -  i         * dx) / dx) *
                               kernel((y - (j + 0.5f) * dx) / dx) / dx;
                float grad_y = kernel((x -  i         * dx) / dx) *
                                kernel_grad((y - (j + 0.5f) * dx) / dx) / dx;

                float affine = B10 * grad_x + B11 * grad_y;

                float velocity = kern * v;
#pragma omp atomic
                vy->values[j * nx + i] += mp * (velocity + affine);

#pragma omp atomic
                mass_y->values[j * nx + i] += mp * kern;
            }
        }
    }

    // Divide momentum by mass to get velocity
#pragma omp parallel for collapse(2)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            float mx = GET(mass_x, i, j);
            float my = GET(mass_y, i, j);
            if (mx > 0.0f) SET(vx, i, j, GET(vx, i, j) / mx);
            if (my > 0.0f) SET(vy, i, j, GET(vy, i, j) / my);
        }
    }

    return EXIT_SUCCESS;
}

inline int grid_to_particles(
    particle_field *particles,
    scalar_field *vx, scalar_field *vy, Metrics &m, std::ofstream &log_file)
{
    LOG_INFO(log_file, "APIC G->P transfer");

    int nx = vx->nx, ny = vx->ny;
    float dx = vx->dx;

#pragma omp parallel for
    for (int k = 0; k < particles->N; k++) {
        float x = particles->xyz[2 * k];
        float y = particles->xyz[2 * k + 1];

        float new_u = 0, new_v = 0;

        // bx = row vector, Dx = 2x2 matrix (x-faces only)
        float bx0 = 0, bx1 = 0;

        // by = row vector, Dy = 2x2 matrix (y-faces only)
        float by0 = 0, by1 = 0;

        // --- x-faces ---
        int i_vx = std::max(0, (int)(x / dx - 0.5)); // matches get_speed
        int j_vx = std::max(0, (int)(y / dx));
        for (int j = j_vx; j < std::min(ny, j_vx + 2); j++) {
            for (int i = i_vx; i < std::min(nx, i_vx + 2); i++) {

                float kern = kernel((x - (i + 0.5f) * dx) / dx) *
                             kernel((y -  j          * dx) / dx);
                
                float ox = (i + 0.5f) * dx - x;
                float oy =  j         * dx - y;

                float vi = GET(vx, i, j);

                new_u += kern * vi;

                bx0 += kern * vi * ox;
                bx1 += kern * vi * oy;
            }
        }

        // --- y-faces ---
        int i_vy = std::max(0, (int)(x / dx));
        int j_vy = std::max(0, (int)(y / dx - 0.5f));
        for (int j = j_vy; j < std::min(ny, j_vy + 2); j++) {
            for (int i = i_vy; i < std::min(nx, i_vy + 2); i++) {

                float kern = kernel((x -  i         * dx) / dx) *
                             kernel((y - (j + 0.5f) * dx) / dx);
                
                float ox =  i         * dx - x;
                float oy = (j + 0.5f) * dx - y;

                float vi = GET(vy, i, j);
                new_v += kern * vi;

                // by += w * vy * o  (vector)
                by0 += kern * vi * ox;
                by1 += kern * vi * oy;
            }
        }

        particles->velocity[2 * k]     = new_u;
        particles->velocity[2 * k + 1] = new_v;

        particles->B[4*k]     = bx0 ;  // C00
        particles->B[4*k + 1] = bx1 ;  // C01
        particles->B[4*k + 2] = by0;  // C10
        particles->B[4*k + 3] = by1;  // C11

        m.singularity_count = 0;
    }
    return EXIT_SUCCESS;
}  */

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
    build_thermal_bc(therm_bcs, data, log_file);

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
    write_manifest_vtk(T->name, dt, nt, sampling_rate, 1, 0, log_file,
                       work_dir);

    // Initial state
    write_scalar_vtk(vx, 0, 0, log_file, work_dir);
    write_scalar_vtk(vy, 0, 0, log_file, work_dir);
    write_scalar_vtk(p, 0, 0, log_file, work_dir);
    write_scalar_vtk(div, 0, 0, log_file, work_dir);
    write_scalar_vtk(dom, 0, 0, log_file, work_dir);
    write_scalar_vtk(T, 0, 0, log_file, work_dir);

    std::vector<int> density(nx * ny, 0);

    // Main time loop
    // bool inverted = false;
    bool first_loop = true;
    std::cout << "Starting simulation" << std::endl;

    Metrics m;
    m = initialize_metrics(m, log_file);
    std::vector<std::string> headers = build_headers(data["metrics"]);
    m = compute_metrics(dom, p, vx, vy, div, dx, 0, nt, m, data["metrics"], log_file);
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

        t7 = std::chrono::high_resolution_clock::now();

        advect(particles, vx, vy, dt, log_file);

        std::fill(density.begin(), density.end(), 0);
        check_particles(particles, dom, m, density, log_file);
        refill_domain(particles, dom, vx, vy, T, density, particle_density, refill,
                     creation_rate, dt, rng, m, log_file);
        
        t2 = std::chrono::high_resolution_clock::now();

        if (gravity)
            apply_gravity(particles, g, dt, beta, T0);

        particles_to_grid(particles, mass, vx, vy, mass_x, mass_y, log_file);
        particles_temp_to_grid(particles, T, kern_sum_T, log_file);

        particles_to_grid(particles, mass, vx, vy, mass_x, mass_y,
                                log_file);
        
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

        apply_thermal_eq(T, T_temp, therm_bcs, dt, c, rho, k, tol, max_iter,
                         log_file);

        project_velocity(p, vx, vy, dom, dx, dt, rho, log_file,
                         speed_condition);

        t4 = std::chrono::high_resolution_clock::now();

        // This is to be able to save it. It serves no purpose in the
        // algorithm
        divergence(vx, vy, div, dom, speed_condition, log_file);

        grid_to_particles(particles, vx, vy, m, log_file);

        t5 = std::chrono::high_resolution_clock::now();
        
        // save files
        if (sampling_rate && !(i % sampling_rate)) {
            write_scalar_vtk(vx, i, 0, log_file, work_dir);
            write_scalar_vtk(vy, i, 0, log_file, work_dir);
            write_scalar_vtk(p, i, 0, log_file, work_dir);
            write_scalar_vtk(div, i, 0, log_file, work_dir);
            write_scalar_vtk(dom, i, 0, log_file, work_dir);
            write_scalar_vtk(T, i, 0, log_file, work_dir);

            write_particles_vtp(particles, i, 0, 2, log_file, work_dir);
        }

        m = compute_metrics(dom, p, vx, vy, div, dx, i, nt, m, data["metrics"], log_file);
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