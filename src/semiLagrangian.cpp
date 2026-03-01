
#include "conditions.hpp"
#include "data.hpp"
#include "nlohmann/json.hpp"
#include "utils.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
using json = nlohmann::json;
/*
 @brief checks the validity of parameters (except boundary and initial
 conditions)
 @param data: the whole data json
 @param log_file: the log file
 TODO: expand this with the new options
*/
int check_params(json &data, std::ofstream &log_file) {
    LOG_INFO(log_file, "Checking the input parameters");
    // grid
    if (data.contains("grid")) {
        if (data["grid"].type() != json::value_t::array) {
            LOG_ERR(log_file, "\"grid\" not provided as array");
            return EXIT_FAILURE;
        }
        if (data["grid"].size() != 2) {
            LOG_ERR(log_file, "\"grid\" not 2 elements long");
            return EXIT_FAILURE;
        }
        if (data["grid"][0] <= 0 || data["grid"][1] <= 0) {
            LOG_ERR(log_file,
                    "Elements from \"grid\" cannot be zero or negative");
            return EXIT_FAILURE;
        }
    }
    // space_steps
    if (data.contains("space_steps")) {
        if (data["space_steps"].type() != json::value_t::number_float) {
            LOG_ERR(log_file, "\"space_steps\" not provided as float");
            return EXIT_FAILURE;
        }
        if (data["space_steps"] <= 0) {
            LOG_ERR(log_file, "\"space_steps\" cannot be zero or negative");
            return EXIT_FAILURE;
        }
    }
    // Time
    if (data.contains("nt")) {
        if (data["nt"].type() != json::value_t::number_unsigned) {
            LOG_ERR(log_file, "\"nt\" not provided as int");
            return EXIT_FAILURE;
        }
        if (data["nt"] <= 0) {
            LOG_ERR(log_file, "\"nt\" cannot be zero or negative");
            return EXIT_FAILURE;
        }
    }
    if (data.contains("delta_t")) {
        if (data["delta_t"].type() != json::value_t::number_float) {
            LOG_ERR(log_file, "\"delta_t\" not provided as float");
            return EXIT_FAILURE;
        }
        if (data["delta_t"] <= 0) {
            LOG_ERR(log_file, "\"delta_t\" cannot be zero or negative");
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
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
/*
 @brief computes the divergence of the velocity field
 @param vx, vy: the velocity field
 @param div: the divergence field
 @param dx: the grid spacing
 @return the maximum divergence in the field, for logging purposes
*/
inline int divergence(scalar_field *vx, scalar_field *vy, scalar_field *div,
                      std::ofstream &log_file) {
    LOG_INFO(log_file, "Computing the divergence");
    int nx = div->nx;
    int ny = div->ny;
    float dx = div->dx;
#pragma omp parallel for collapse(2)
    // div has the size of pressure so even when i = nx-1 or j = ny-1, we can
    // safely access vx and vy
    for (int j = 1; j < ny; j++) {
        for (int i = 1; i < nx; i++) {
            float dudx = (GET(vx, i, j) - GET(vx, i - 1, j)) / dx;
            float dvdy = (GET(vy, i, j) - GET(vy, i, j - 1)) / dx;
            float d = dudx + dvdy;
            SET(div, i, j, d);
        }
    }
    int j = 0;
    for (int i = 1; i < nx; i++) {
        float dudx = (GET(vx, i, j) - GET(vx, i - 1, j)) / dx;
        float dvdy = (GET(vy, i, j)) / dx;
        float d = dudx + dvdy;
        SET(div, i, j, d);
    }
    int i = 0;
    for (int j = 1; j < ny; j++) {
        float dudx = (GET(vx, i, j)) / dx;
        float dvdy = (GET(vy, i, j) - GET(vy, i, j - 1)) / dx;
        float d = dudx + dvdy;
        SET(div, i, j, d);
    }
    float dudx = (GET(vx, 0, 0)) / dx;
    float dvdy = (GET(vy, 0, 0)) / dx;
    float d = dudx + dvdy;
    SET(div, 0, 0, d);
    return EXIT_SUCCESS;
}
/*
 @brief solves the Poisson equation for the pressure using Jacobi
 iterations
 @param p: the pressure field
 @param temp_p: a temporary pressure field needed to work
 @param div: the divergence field
 @param vx, vy: the velocity field
 @param dom: the domain field
 @param tol: the tolerance at which to stop
 @param dt: the time step
 @param rho: the density
 @param max_iter: the max number of iterations
 @param first_looop: whether this is the first time loop or not
*/
inline int jacobi(scalar_field *p, scalar_field *temp_p, scalar_field *div,
                  scalar_field *vx, scalar_field *vy, scalar_field *dom,
                  float tol, float dt, float rho, int max_iter, bool first_loop,
                  std::ofstream &log_file) {
    LOG_INFO(log_file, "Starting Jacobi");
    int nx = p->nx;
    int ny = p->ny;
    float dx = dom->dx;
    const float alpha = dx * dx * rho / dt;
    float beta = rho * dx / dt;
    float maxPdiff = 1.0;
    int iter = 0;
    // as it's the one that has problems converging, putting this really high
    if (first_loop) {
        max_iter = nx * ny;
    }
    bool inverted = false;
    while (maxPdiff >
               tol * std::max(*std::max_element(p->values, p->values + nx * ny),
                              1.0f) &&
           iter < max_iter) {
        maxPdiff = 0.0;
#pragma omp parallel for collapse(2) reduction(max : maxPdiff)
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                if (GET(dom, i, j) == SOLID)
                    continue;
                if (GET(dom, i, j) == AIR)
                    continue;

                float dom_left = GET(dom, i - 1, j);
                float p_left = 0.0f;
                if (dom_left == LIQUID || dom_left == AIR) {
                    p_left = GET(p, i - 1, j);
                } else if (dom_left == SOLID) {
                    p_left = GET(p, i, j) - beta * GET(vx, i - 1, j);
                }

                float dom_right = GET(dom, i + 1, j);
                float p_right = 0.0f;
                if (dom_right == LIQUID || dom_right == AIR) {
                    p_right = GET(p, i + 1, j);
                } else if (dom_right == SOLID) {
                    p_right = GET(p, i, j) + beta * GET(vx, i, j);
                }

                float dom_down = GET(dom, i, j - 1);
                float p_down = 0.0f;
                if (dom_down == LIQUID || dom_down == AIR) {
                    p_down = GET(p, i, j - 1);
                } else if (dom_down == SOLID) {
                    p_down = GET(p, i, j) - beta * GET(vy, i, j - 1);
                }

                float dom_up = GET(dom, i, j + 1);
                float p_up = 0.0f;
                if (dom_up == LIQUID || dom_up == AIR) {
                    p_up = GET(p, i, j + 1);
                } else if (dom_up == SOLID) {
                    p_up = GET(p, i, j) + beta * GET(vy, i, j);
                }

                float new_p = (p_left + p_right + p_down + p_up -
                               alpha * GET(div, i, j)) /
                              4.0;
                float Pdiff = std::abs(new_p - GET(p, i, j));
                maxPdiff = std::max(maxPdiff, Pdiff);
                SET(temp_p, i, j, new_p);
            }
        }
        // removing the mean of the pressure, so that we can use Neumann conds
        // only, instead of having to fix the pressure somewhere
        float sum = 0;
        for (int j = 1; j < ny - 1; j++)
            for (int i = 1; i < nx - 1; i++)
                sum += GET(temp_p, i, j);
        float mean = sum / (nx * ny);
        for (int j = 1; j < ny - 1; j++)
            for (int i = 1; i < nx - 1; i++)
                SET(temp_p, i, j, GET(temp_p, i, j) - mean);
        // as we updated temp_p, we need to invert
        scalar_field *tmp = p;
        p = temp_p;
        temp_p = tmp;
        inverted = !inverted;
        iter++;
    }
    if (iter == max_iter)
        LOG_WARN(log_file, "Jacobi stopped at " << max_iter << " iterations")
    else
        LOG_INFO(log_file, "Jacobi converged in " << iter << " iterations");
    LOG_INFO(log_file, "Max pressure difference: " << maxPdiff);
    // to make sure that p has the right information for the rest of the time
    // loop
    if (inverted) {
        scalar_field *tmp = p;
        p = temp_p;
        temp_p = tmp;
    }
    return EXIT_SUCCESS;
}
/*
 @brief solves the Poisson equation for the pressure using SOR
 iterations
 @param p: the pressure field
 @param div: the divergence field
 @param vx, vy: the velocity field
 @param dom: the domain field
 @param tol: the tolerance at which to stop
 @param dt: the time step
 @param rho: the density
 @param max_iter: the max number of iterations
*/
inline int sor(scalar_field *p, scalar_field *div, scalar_field *vx,
               scalar_field *vy, scalar_field *dom, float tol, float dt,
               float rho, int max_iter, std::ofstream &log_file) {
    LOG_INFO(log_file, "Starting SOR")
    int nx = p->nx;
    int ny = p->ny;
    float dx = dom->dx;
    const float alpha = dx * dx * rho / dt;
    float beta = rho * dx / dt;
    float maxPdiff = 1.0;
    int iter = 0;
    // parameters needed for the algorithm
    const int N = std::min(nx, ny);
    const float pi = 3.14159265358979;
    const float omega = std::min(1.95f, 2.0f / (1.0f + std::sin(pi / N)));
    while (maxPdiff >
               tol * std::max(*std::max_element(p->values, p->values + nx * ny),
                              1.0f) &&
           iter < max_iter) {
        maxPdiff = 0.0;
        // to be able to parallelize, need checkered grids
        for (int color = 0; color < 2; color++) {
#pragma omp parallel for collapse(2) reduction(max : maxPdiff)
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    if ((i + j) % 2 != color)
                        continue;
                    if (GET(dom, i, j) == SOLID || GET(dom, i, j) == AIR){
                        continue;
                    }

                    float dom_left = GET(dom, i - 1, j);
                    float p_left = 0.0f;
                    if (dom_left == LIQUID) {
                        p_left = GET(p, i - 1, j);
                    } else if (dom_left == SOLID) {
                        p_left = GET(p, i, j) - beta * GET(vx, i - 1, j);
                    } else if (dom_left == AIR) {
                        p_left = 0.0f;
                    }

                    float dom_right = GET(dom, i + 1, j);
                    float p_right = 0.0f;
                    if (dom_right == LIQUID) {
                        p_right = GET(p, i + 1, j);
                    } else if (dom_right == SOLID) {
                        p_right = GET(p, i, j) + beta * GET(vx, i, j);
                    } else if (dom_right == AIR) {
                        p_right = 0.0f;
                    }

                    float dom_down = GET(dom, i, j - 1);
                    float p_down = 0.0f;
                    if (dom_down == LIQUID) {
                        p_down = GET(p, i, j - 1);
                    } else if (dom_down == SOLID) {
                        p_down = GET(p, i, j) - beta * GET(vy, i, j - 1);
                    } else if (dom_down == AIR) {
                        p_down = 0.0f;
                    }

                    float dom_up = GET(dom, i, j + 1);
                    float p_up = 0.0f;
                    if (dom_up == LIQUID) {
                        p_up = GET(p, i, j + 1);
                    } else if (dom_up == SOLID) {
                        p_up = GET(p, i, j) + beta * GET(vy, i, j);
                    } else if (dom_up == AIR) {
                        p_up = 0.0f;
                    }

                    float new_p = (p_left + p_right + p_down + p_up -
                                   alpha * GET(div, i, j)) /
                                  4.0;
                    float Pdiff = std::abs(new_p - GET(p, i, j));
                    maxPdiff = std::max(maxPdiff, Pdiff);
                    SET(p, i, j, GET(p, i, j) + omega * (new_p - GET(p, i, j)));
                }
            }
        }
        /* // Need to remove the mean pressure, so that we have a condition in
        // addition to the Neumann ones
        float sum = 0;
#pragma omp parallel for collapse(2) reduction(+ : sum)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
                sum += GET(p, i, j);
        float mean = sum / (nx * ny);
#pragma omp parallel for collapse(2)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
                SET(p, i, j, GET(p, i, j) - mean); */
        iter++;
    }
    if (iter == max_iter)
        LOG_WARN(log_file, "SOR stopped at " << max_iter << " iterations")
    else
        LOG_INFO(log_file, "SOR converged in " << iter << " iterations");
    LOG_INFO(log_file, "Max pressure difference: " << maxPdiff);
    return EXIT_SUCCESS;
}
/*
 @brief projects the velocity field to make it divergence free
 @param p: the pressure field
 @param vx, vy: the velocity field
 @param dt: the time step
 @param rho: the density
*/
inline int project_velocity(scalar_field *p, scalar_field *vx, scalar_field *vy,
                            scalar_field *dom, float dx, float dt, float rho,
                            std::ofstream &log_file) {
    LOG_INFO(log_file, "Projecting the velocity field")
    int vx_nx = vx->nx;
    int vx_ny = vx->ny;
#pragma omp parallel for collapse(2)
    for (int j = 0; j < vx_ny; j++) {
        for (int i = 0; i < vx_nx; i++) {
            if (GET(dom, i+1, j) == SOLID ||
                GET(dom, i, j) == SOLID) {
                SET(vx, i, j, 0.0);
                continue;
            }
            if (i == vx_nx - 1 && GET(dom,i,j) == AIR) {
                float v_x = GET(vx, i-1, j);
                SET(vx, i, j, v_x);
                continue;
            }
            float gradp_x = (GET(p, i + 1, j) - GET(p, i, j)) / dx;
            SET(vx, i, j, GET(vx, i, j) - dt * gradp_x / rho);
        }
    }
    int vy_nx = vy->nx;
    int vy_ny = vy->ny;
#pragma omp parallel for collapse(2)
    for (int j = 0; j < vy_ny; j++) {
        for (int i = 0; i < vy_nx; i++) {
            if (GET(dom, i, j + 1) == SOLID ||
                GET(dom, i, j) == SOLID) {
                SET(vy, i, j, 0.0);
                continue;
            }
            if (j == vy_ny - 1 && GET(dom,i,j) == AIR) {
                float v_y = GET(vy, i, j-1);
                SET(vy, i, j, v_y);
                continue;
            }
            float gradp_y = (GET(p, i, j + 1) - GET(p, i, j)) / dx;
            SET(vy, i, j, GET(vy, i, j) - dt * gradp_y / rho);
        }
    }
    return EXIT_SUCCESS;
}
/*
 @brief the semi lagrangian solver
 @param data: the whole json
 @param log_file: the log file
*/
int solver_semi_lagrangian(json &data, std::ofstream &log_file) {
    LOG_INFO(log_file, "Starting the semi-lagrangian solver");
    auto t0 = std::chrono::high_resolution_clock::now();
    if (check_params(data, log_file)) {
        LOG_ERR(log_file, "Problem checking the input parameters")
        return EXIT_FAILURE;
    }
    // Getting base params
    const unsigned int nx = data["grid"][0], ny = data["grid"][1];
    const float dx = data["space_steps"];
    const int sampling_rate = data["sampling_rate"];
    float dt = 0.1;
    if (data.contains("delta_t"))
        dt = data["delta_t"];
    unsigned int nt = 10;
    if (data.contains("nt"))
        nt = data["nt"];
    float rho = 1.0;
    if (data.contains("rho"))
        rho = data["rho"];
    float tol = 1e-5;
    if (data.contains("tol"))
        tol = data["tol"];
    int max_iter = 1e5;
    if (data.contains("max_iter"))
        max_iter = data["max_iter"];
    // Initialising the fields
    scalar_field *vx =
        scalar_field_init("vx", nx + 1, ny, 0.5, 0, dx, log_file);
    scalar_field *vy =
        scalar_field_init("vy", nx, ny + 1, 0, 0.5, dx, log_file);
    scalar_field *p = scalar_field_init("p", nx, ny, 0, 0, dx, log_file);
    scalar_field *div = scalar_field_init("div", nx, ny, 0, 0, dx, log_file);
    scalar_field *dom =
        scalar_field_init("dom", nx, ny, 0, 0, dx, log_file);
    if (!vx || !vy || !p || !div || !dom) {
        LOG_ERR(log_file, "An error occured initializing fields.");
        return EXIT_FAILURE;
    }
    // Applying the initial conditions
    initialize_domain(dom, data, "ic_cell", log_file);
    initialize_speed(vx, dom, data, "ic_vx", log_file);
    initialize_speed(vy, dom, data, "ic_vy", log_file);
    scalar_field *temp_vx = scalar_field_copy(vx, log_file);
    scalar_field *temp_vy = scalar_field_copy(vy, log_file);
    scalar_field *temp_p = scalar_field_copy(p, log_file);
    write_scalar_vtk(vx, 0, 0, log_file);
    write_scalar_vtk(vy, 0, 0, log_file);
    write_scalar_vtk(p, 0, 0, log_file);
    write_scalar_vtk(div, 0, 0, log_file);
    // Main time loop
    bool inverted = false;
    bool first_loop = true;
    for (unsigned int i = 1; i < nt; i++) {
        log_file << "\n";
        LOG_INFO(log_file, "Starting time loop " << i << " out of " << nt);
        divergence(vx, vy, div, log_file);
        // Making sure that the mean of the divergence is zero
        // Comes from an integral condition to have a solution
        float sum = 0;
        for (unsigned int j = 0; j < ny; j++)
            for (unsigned int i = 0; i < nx; i++)
                sum += GET(div, i, j);
        LOG_INFO(log_file, "Total divergence: " << sum);
        if (data["iteration_algo"] == "Jacobi")
            jacobi(p, temp_p, div, vx, vy, dom, tol, dt, rho, max_iter,
                   first_loop, log_file);
        else if (data["iteration_algo"] == "SOR")
            sor(p, div, vx, vy, dom, tol, dt, rho, max_iter, log_file);
        else {
            LOG_ERR(log_file, "Iteration algorithm not supported");
            return EXIT_FAILURE;
        }
        project_velocity(p, vx, vy, dom, dx, dt, rho, log_file);
        // This is to be able to save it. It serves no purpose in the algorithm
        divergence(vx, vy, div, log_file);
        // save files, when the divergence is zero
        if (sampling_rate && !(i % sampling_rate)) {
            write_scalar_vtk(vx, i, 0, log_file);
            write_scalar_vtk(vy, i, 0, log_file);
            write_scalar_vtk(p, i, 0, log_file);
            write_scalar_vtk(div, i, 0, log_file);
        }
        // advect
        advect(vx, vy, dt, vx, temp_vx, log_file);
        advect(vx, vy, dt, vy, temp_vy, log_file);
        inverted = !inverted;
        scalar_field *invert_vx = vx;
        scalar_field *invert_vy = vy;
        vx = temp_vx;
        vy = temp_vy;
        temp_vx = invert_vx;
        temp_vy = invert_vy;
        first_loop = false;
        // Resetting the pressure field
#pragma omp parallel for collapse(2)
        for (unsigned int j = 0; j < ny; j++)
            for (unsigned int i = 0; i < nx; i++)
                SET(p, i, j, 0.0f);
    }
    write_manifest_vtk(vx->name, dt, nt, sampling_rate, 1, 0, log_file);
    write_manifest_vtk(vy->name, dt, nt, sampling_rate, 1, 0, log_file);
    write_manifest_vtk(p->name, dt, nt, sampling_rate, 1, 0, log_file);
    write_manifest_vtk(div->name, dt, nt, sampling_rate, 1, 0, log_file);
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
    return EXIT_SUCCESS;
}