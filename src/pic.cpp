#include "conditions.hpp"
#include "data.hpp"
#include "nlohmann/json.hpp"
#include "utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <random>

using json = nlohmann::json;

/*
 @brief checks the validity of parameters (except boundary and initial
 conditions)
 @param data: the whole data json
 @param log_file: the log file
 TODO: expand this with the new options
*/
int check_params_pic(json &data, std::ofstream &log_file) {
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

inline int get_speed(float *v_x, float *v_y, float x, float y, scalar_field *vx,
                     scalar_field *vy, std::ofstream &log_file) {
    float dx = vx->dx;
    // vx
    *v_x = 0;
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
        LOG_WARN(log_file, "y_1 too big, y = " << y << " and y_1 = " << y_1)
    y_2 = y_1 + 1;
    y_2 = std::min(vx->ny - 1, y_2);

    *v_x = interpolate_bilinear(x, y, (x_1 + vx->x_internal) * dx,
                                (y_1 + vx->y_internal) * dx, GET(vx, x_1, y_1),
                                GET(vx, x_2, y_1), GET(vx, x_1, y_2),
                                GET(vx, x_2, y_2), dx, dx);

    // vy
    *v_y = 0;
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

    *v_y = interpolate_bilinear(x, y, (x_1 + vy->x_internal) * dx,
                                (y_1 + vy->y_internal) * dx, GET(vy, x_1, y_1),
                                GET(vy, x_2, y_1), GET(vy, x_1, y_2),
                                GET(vy, x_2, y_2), dx, dx);

    return EXIT_SUCCESS;
}

/*
 @brief avects the particles based on their velocity
 @
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

inline float kernel(float r) {
    if (0 <= r && r <= 1)
        return 1 - r;
    if (-1 <= r && r <= 0)
        return 1 + r;
    return 0;
}

inline int particles_speed_to_grid(particle_field *particles, scalar_field *vx,
                                   scalar_field *vy, scalar_field *kern_sum,
                                   std::ofstream &log_file) {
    LOG_INFO(log_file, "Transferring the speed of particles to the grid");

    int nx = vy->nx, ny = vx->ny;
    // int S = nx * ny;
    // float W = (float)particles->N / (float)S;
    float dx = vx->dx;

#pragma omp parallel for collapse(2)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            SET(vx, i, j, 0);
            SET(vy, i, j, 0);
            SET(kern_sum, i, j, 0);
        }
    }

#pragma omp parallel for
    for (int k = 0; k < particles->N; k++) {
        float x = particles->xyz[2 * k];
        float y = particles->xyz[2 * k + 1];
        float u = particles->velocity[2 * k];
        float v = particles->velocity[2 * k + 1];

        for (int j = std::max(0, (int)(y / dx - 1)); j < y / dx + 1 && j < ny;
             j++) {
            for (int i = std::max(0, (int)(x / dx - 1));
                 i < x / dx + 1 && i < nx; i++) {
                float dist_x = x - i * dx;
                float dist_y = y - j * dx;
                float kern = kernel(dist_x / dx) * kernel(dist_y / dx);

#pragma omp critical
                {
                    float vx_pre = GET(vx, i, j);
                    float vy_pre = GET(vy, i, j);
                    float kern_pre = GET(kern_sum, i, j);
                    SET(vx, i, j, vx_pre + u * kern);
                    SET(vy, i, j, vy_pre + v * kern);
                    SET(kern_sum, i, j, kern_pre + kern);
                }
            }
        }
    }

#pragma omp parallel for collapse(2)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            float kern = GET(kern_sum, i, j);
            if (kern != 0) {
                SET(vx, i, j, GET(vx, i, j) / kern);
                SET(vy, i, j, GET(vy, i, j) / kern);
            }
        }
    }

    return EXIT_SUCCESS;
}

inline int grid_speed_to_particles(particle_field *particles, scalar_field *vx,
                                   scalar_field *vy, std::ofstream &log_file) {

    for (int k = 0; k < particles->N; k++) {
        float x = particles->xyz[2 * k];
        float y = particles->xyz[2 * k + 1];

        float v_x, v_y;
        get_speed(&v_x, &v_y, x, y, vx, vy, log_file);

        particles->velocity[2 * k] = v_x;
        particles->velocity[2 * k + 1] = v_y;
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
inline int advect_pic_old(scalar_field *vx, scalar_field *vy, float dt,
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

            float v_x, v_y;
            get_speed(&v_x, &v_y, x, y, vx, vy, log_file);

            // xp
            float xp_x = x - dt * v_x;
            float xp_y = y - dt * v_y;
            xp_x = std::max((float)0, xp_x);
            xp_y = std::max((float)0, xp_y);
            xp_x = std::min((q_n->nx - 1) * dx, xp_x);
            xp_y = std::min((q_n->ny - 1) * dx, xp_y);

            int x_1 = 0, y_1 = 0, x_2 = 0, y_2 = 0;
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
inline int divergence_pic(scalar_field *vx, scalar_field *vy, scalar_field *div,
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
 @brief computes the residual term of the pressure computation
 @params scarlar_field res: where the resulting Ax term will be stored
 @params same as everywhere
 @returns: the 2-norm of the residual
*/
inline float residual_pic(scalar_field *p, scalar_field *vx, scalar_field *vy,
                          scalar_field *div, scalar_field *dom, float rho,
                          float dt, std::ofstream &log_file) {
    // LOG_INFO(log_file, "Computing the residual");
    if (0)
        log_file << "dummy";

    int nx = p->nx;
    int ny = p->ny;
    float dx = dom->dx;
    const float alpha = dx * dx * rho / dt;
    float beta = rho * dx / dt;
    float norm_squared = 0;

#pragma omp parallel for collapse(2) reduction(+ : norm_squared)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if (GET(dom, i + 1, j + 1) == SOLID) {
                continue;
            }

            float dom_left = GET(dom, i, j + 1);
            float p_left = 0;
            // need to extrapolate the speed
            if (i == 0) {
                // p_left = GET(p, i, j) -
                //          beta * (2 * GET(vx, i, j) - GET(vx, i + 1, j));
                p_left = GET(p, i, j);
            } else if (dom_left == LIQUID) {
                p_left = GET(p, i - 1, j);
            } else if (dom_left == SOLID) {
                p_left = GET(p, i, j) - beta * GET(vx, i - 1, j);
            }

            float dom_right = GET(dom, i + 2, j + 1);
            float p_right = 0;
            // no need to extrapolate due to convention
            if (i == nx - 1) {
                p_right = GET(p, i, j);
            } else if (dom_right == LIQUID) {
                p_right = GET(p, i + 1, j);
            } else if (dom_right == SOLID) {
                p_right = GET(p, i, j) + beta * GET(vx, i, j);
            }

            float dom_down = GET(dom, i + 1, j);
            float p_down = 0;
            if (j == 0) {
                // p_down = GET(p, i, j) -
                //          beta * (2 * GET(vy, i, j) - GET(vy, i, j + 1));
                p_down = GET(p, i, j);
            } else if (dom_down == LIQUID) {
                p_down = GET(p, i, j - 1);
            } else if (dom_down == SOLID) {
                p_down = GET(p, i, j) - beta * GET(vy, i, j - 1);
            }

            float dom_up = GET(dom, i + 1, j + 2);
            float p_up = 0;
            // no need to extrapolate due to convention
            if (j == ny - 1) {
                p_up = GET(p, i, j);
            } else if (dom_up == LIQUID) {
                p_up = GET(p, i, j + 1);
            } else if (dom_up == SOLID) {
                p_up = GET(p, i, j) + beta * GET(vy, i, j);
            }

            float residue = GET(p, i, j) - (p_left + p_right + p_down + p_up -
                                            alpha * GET(div, i, j)) /
                                               4.0;

            norm_squared += residue * residue;
        }
    }

    return std::sqrt(norm_squared);
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
inline int jacobi_pic(scalar_field *p, scalar_field *temp_p, scalar_field *div,
                      scalar_field *vx, scalar_field *vy, scalar_field *dom,
                      float tol, float dt, float rho, int max_iter,
                      bool first_loop, std::ofstream &log_file) {
    LOG_INFO(log_file, "Starting Jacobi");

    int nx = p->nx;
    int ny = p->ny;
    float dx = dom->dx;
    const float alpha = dx * dx * rho / dt;
    float beta = rho * dx / dt;
    int iter = 0;

    // as it's the one that has problems converging, putting this really
    // high
    if (first_loop) {
        max_iter = nx * ny;
    }

    float norm_b = 0;
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            norm_b += alpha * alpha * GET(div, i, j) * GET(div, i, j);
        }
    }
    norm_b = std::sqrt(norm_b);
    norm_b += 1e-7;

    float residue = residual_pic(p, vx, vy, div, dom, rho, dt, log_file);
    float condition = residue / (norm_b);

    bool inverted = false;

    while (condition > tol && iter < max_iter) {
        residue = 0;
#pragma omp parallel for collapse(2) reduction(+ : residue)
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                if (GET(dom, i + 1, j + 1) == SOLID)
                    continue;

                float dom_left = GET(dom, i, j + 1);
                float p_left = 0;
                // need to extrapolate the speed
                if (i == 0) {
                    // p_left = GET(p, i, j) -
                    //          beta * (2 * GET(vx, i, j) - GET(vx, i + 1,
                    //          j));
                    p_left = GET(p, i, j);
                } else if (dom_left == LIQUID) {
                    p_left = GET(p, i - 1, j);
                } else if (dom_left == SOLID) {
                    p_left = GET(p, i, j) - beta * GET(vx, i - 1, j);
                }

                float dom_right = GET(dom, i + 2, j + 1);
                float p_right = 0;
                // no need to extrapolate due to convention
                if (i == nx - 1) {
                    p_right = GET(p, i, j);
                } else if (dom_right == LIQUID) {
                    p_right = GET(p, i + 1, j);
                } else if (dom_right == SOLID) {
                    p_right = GET(p, i, j) + beta * GET(vx, i, j);
                }

                float dom_down = GET(dom, i + 1, j);
                float p_down = 0;
                if (j == 0) {
                    // p_down = GET(p, i, j) -
                    //          beta * (2 * GET(vy, i, j) - GET(vy, i, j +
                    //          1));
                    p_down = GET(p, i, j);
                } else if (dom_down == LIQUID) {
                    p_down = GET(p, i, j - 1);
                } else if (dom_down == SOLID) {
                    p_down = GET(p, i, j) - beta * GET(vy, i, j - 1);
                }

                float dom_up = GET(dom, i + 1, j + 2);
                float p_up = 0;
                // no need to extrapolate due to convention
                if (j == ny - 1) {
                    p_up = GET(p, i, j);
                } else if (dom_up == LIQUID) {
                    p_up = GET(p, i, j + 1);
                } else if (dom_up == SOLID) {
                    p_up = GET(p, i, j) + beta * GET(vy, i, j);
                }

                float new_p = (p_left + p_right + p_down + p_up -
                               alpha * GET(div, i, j)) /
                              4.0;
                residue += (GET(p, i, j) - new_p) * (GET(p, i, j) - new_p);

                SET(temp_p, i, j, new_p);
            }
        }

        // removing the mean of the pressure, so that we can use Neumann
        // conds only, instead of having to fix the pressure somewhere
        float sum = 0;
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
                sum += GET(temp_p, i, j);

        float mean = sum / (nx * ny);
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
                SET(temp_p, i, j, GET(temp_p, i, j) - mean);

        // as we updated temp_p, we need to invert
        scalar_field *tmp = p;
        p = temp_p;
        temp_p = tmp;
        inverted = !inverted;

        residue = std::sqrt(residue);
        condition = residue / norm_b;

        iter++;
    }

    if (iter == max_iter)
        LOG_WARN(log_file, "Jacobi stopped at " << max_iter << " iterations")
    else
        LOG_INFO(log_file, "Jacobi converged in " << iter << " iterations");
    LOG_INFO(log_file, "Last residue: " << residue);

    // to make sure that p has the right information for the rest of the
    // time loop
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
inline int sor_pic(scalar_field *p, scalar_field *div, scalar_field *vx,
                   scalar_field *vy, scalar_field *dom, float tol, float dt,
                   float rho, int max_iter, std::ofstream &log_file) {
    LOG_INFO(log_file, "Starting SOR")

    int nx = p->nx;
    int ny = p->ny;
    float dx = dom->dx;
    const float alpha = dx * dx * rho / dt;
    float beta = rho * dx / dt;
    int iter = 0;

    // parameters needed for the algorithm
    const int N = std::min(nx, ny);
    const float pi = 3.14159265358979;
    const float omega = std::min(1.95f, 2.0f / (1.0f + std::sin(pi / N)));

    float norm_b = 0;
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            norm_b += alpha * alpha * GET(div, i, j) * GET(div, i, j);
        }
    }
    norm_b = std::sqrt(norm_b);
    norm_b += 1e-7;

    float residue = residual_pic(p, vx, vy, div, dom, rho, dt, log_file);
    float condition = residue / (norm_b);

    // while (maxPdiff >
    //            tol * std::max(*std::max_element(p->values, p->values + nx *
    //            ny),
    //                           1.0f) &&
    //        iter < max_iter) {
    while ((condition > tol) && iter < max_iter) {
        residue = 0;
        // to be able to parallelize, need checkered grids
        for (int color = 0; color < 2; color++) {

#pragma omp parallel for collapse(2) reduction(+ : residue)
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    if ((i + j) % 2 != color)
                        continue;

                    if (GET(dom, i + 1, j + 1) == SOLID)
                        continue;

                    float dom_left = GET(dom, i, j + 1);
                    float p_left = 0;
                    // need to extrapolate the speed
                    if (i == 0) {
                        // p_left = GET(p, i, j) -
                        //          beta * (2 * GET(vx, i, j) - GET(vx, i +
                        //          1, j));
                        p_left = GET(p, i, j);
                    } else if (dom_left == LIQUID) {
                        p_left = GET(p, i - 1, j);
                    } else if (dom_left == SOLID) {
                        p_left = GET(p, i, j) - beta * GET(vx, i - 1, j);
                    }

                    float dom_right = GET(dom, i + 2, j + 1);
                    float p_right = 0;
                    // no need to extrapolate due to convention
                    if (i == nx - 1) {
                        p_right = GET(p, i, j);
                    } else if (dom_right == LIQUID) {
                        p_right = GET(p, i + 1, j);
                    } else if (dom_right == SOLID) {
                        p_right = GET(p, i, j) + beta * GET(vx, i, j);
                    }

                    float dom_down = GET(dom, i + 1, j);
                    float p_down = 0;
                    if (j == 0) {
                        // p_down = GET(p, i, j) -
                        //          beta * (2 * GET(vy, i, j) - GET(vy, i, j
                        //          + 1));
                        p_down = GET(p, i, j);
                    } else if (dom_down == LIQUID) {
                        p_down = GET(p, i, j - 1);
                    } else if (dom_down == SOLID) {
                        p_down = GET(p, i, j) - beta * GET(vy, i, j - 1);
                    }

                    float dom_up = GET(dom, i + 1, j + 2);
                    float p_up = 0;
                    // no need to extrapolate due to convention
                    if (j == ny - 1) {
                        p_up = GET(p, i, j);
                    } else if (dom_up == LIQUID) {
                        p_up = GET(p, i, j + 1);
                    } else if (dom_up == SOLID) {
                        p_up = GET(p, i, j) + beta * GET(vy, i, j);
                    }

                    float new_p = (p_left + p_right + p_down + p_up -
                                   alpha * GET(div, i, j)) /
                                  4.0;
                    residue += (GET(p, i, j) - new_p) * (GET(p, i, j) - new_p);
                    SET(p, i, j, GET(p, i, j) + omega * (new_p - GET(p, i, j)));
                }
            }
        }

        // Need to remove the mean pressure, so that we have a condition in
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
                SET(p, i, j, GET(p, i, j) - mean);

        residue = std::sqrt(residue);
        condition = residue / norm_b;

        iter++;
    }

    if (iter == max_iter)
        LOG_WARN(log_file, "SOR stopped at " << max_iter << " iterations")
    else
        LOG_INFO(log_file, "SOR converged in " << iter << " iterations");
    LOG_INFO(log_file, "Last residue: " << residue);

    return EXIT_SUCCESS;
}

/*
 @brief projects the velocity field to make it divergence free
 @param p: the pressure field
 @param vx, vy: the velocity field
 @param dt: the time step
 @param rho: the density
*/
inline int project_velocity_pic(scalar_field *p, scalar_field *vx,
                                scalar_field *vy, scalar_field *dom, float dx,
                                float dt, float rho, std::ofstream &log_file) {
    LOG_INFO(log_file, "Projecting the velocity field")

    int vx_nx = vx->nx;
    int vx_ny = vx->ny;
    int p_nx = p->nx;

#pragma omp parallel for collapse(2)
    for (int j = 0; j < vx_ny; j++) {
        for (int i = 0; i < vx_nx; i++) {
            if (GET(dom, i + 2, j + 1) == SOLID ||
                GET(dom, i + 1, j + 1) == SOLID) {
                SET(vx, i, j, 0.0);
                continue;
            }
            if (i == p_nx)
                continue;

            float gradp_x = (GET(p, i + 1, j) - GET(p, i, j)) / dx;
            SET(vx, i, j, GET(vx, i, j) - dt * gradp_x / rho);
        }
    }

    int vy_nx = vy->nx;
    int vy_ny = vy->ny;
    int p_ny = p->ny;

#pragma omp parallel for collapse(2)
    for (int j = 0; j < vy_ny; j++) {
        for (int i = 0; i < vy_nx; i++) {
            if (GET(dom, i + 1, j + 2) == SOLID ||
                GET(dom, i + 1, j + 1) == SOLID) {
                SET(vy, i, j, 0.0);
                continue;
            }

            if (j == p_ny)
                continue;

            float gradp_y = (GET(p, i, j + 1) - GET(p, i, j)) / dx;
            SET(vy, i, j, GET(vy, i, j) - dt * gradp_y / rho);
        }
    }

    return EXIT_SUCCESS;
}

inline void initialize_particles_pic(particle_field *particles,
                                     scalar_field *dom, scalar_field *vx,
                                     scalar_field *vy, int density,
                                     std::ofstream &log_file) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0, dom->dx);
    int nx = dom->nx, ny = dom->ny;
    float dx = dom->dx;

    int nb_not_fluid_cases = 0;
    int current_id = 0;

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            float cell = GET(dom, i, j);
            if (cell == LIQUID) {
                for (int k = 0; k < density; k++) {
                    float x = i * dx + dist(gen);
                    float y = j * dx + dist(gen);
                    particles->xyz[2 * current_id] = x;
                    particles->xyz[2 * current_id + 1] = y;

                    float v_x, v_y;
                    get_speed(&v_x, &v_y, x, y, vx, vy, log_file);

                    particles->velocity[2 * current_id] = v_x;
                    particles->velocity[2 * current_id + 1] = v_y;

                    current_id++;
                }

            } else {
                nb_not_fluid_cases++;
            }
        }
    }

    particles->N = current_id;
    particles->xyz.resize(current_id);
    particles->velocity.resize(current_id);
    particles->id.resize(current_id);
}

/*
 @brief the PIC/FLIP
 @param data: the whole json
 @param log_file: the log file
*/
int solver_pic(json &data, std::ofstream &log_file) {
    LOG_INFO(log_file, "Starting the PIC/FLIP solver");

    auto t0 = std::chrono::high_resolution_clock::now();

    if (check_params_pic(data, log_file)) {
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

    int density = data.value("particle_density", 2);

    user_fields fields;
    get_fields(&fields, data, log_file);

    // Initialising the fields
    scalar_field *vx =
        scalar_field_init("vx", nx + 1, ny, 0.5, 0, dx, log_file);
    scalar_field *vy =
        scalar_field_init("vy", nx, ny + 1, 0, 0.5, dx, log_file);
    scalar_field *p = scalar_field_init("p", nx, ny, 0, 0, dx, log_file);
    scalar_field *div = scalar_field_init("div", nx, ny, 0, 0, dx, log_file);
    scalar_field *dom =
        scalar_field_init("dom", nx + 2, ny + 2, 0, 0, dx, log_file);

    scalar_field *kern_sum =
        scalar_field_init("kern_sum", nx, ny, 0, 9, dx, log_file);

    if (!vx || !vy || !p || !div || !dom) {
        LOG_ERR(log_file, "An error occured initializing fields.");
        return EXIT_FAILURE;
    }

    // Applying the initial conditions
    initialize_domain(dom, data, "ic_cell", log_file);
    create_circle(dom, "ic_cylinders", data, log_file);
    initialize_speed(vx, dom, data, "ic_vx", log_file);
    initialize_speed(vy, dom, data, "ic_vy", log_file);

    // scalar_field *temp_vx = scalar_field_copy(vx, log_file);
    // scalar_field *temp_vy = scalar_field_copy(vy, log_file);
    scalar_field *temp_p = scalar_field_copy(p, log_file);

    // Initializing the particles
    particle_field *particles =
        particle_field_init_2D("particles", nx * ny * density, log_file);
    initialize_particles_pic(particles, dom, vx, vy, data["particle_density"],
                             log_file);

    write_manifest_vtk("particles", dt, nt, sampling_rate, 1, 1, log_file);
    write_particles_vtp(particles, 0, 0, 2, log_file);

    write_manifest_vtk(vx->name, dt, nt, sampling_rate, 1, 0, log_file);
    write_manifest_vtk(vy->name, dt, nt, sampling_rate, 1, 0, log_file);
    write_manifest_vtk(p->name, dt, nt, sampling_rate, 1, 0, log_file);
    write_manifest_vtk(div->name, dt, nt, sampling_rate, 1, 0, log_file);
    for (int i = 0; i < fields.nb_fields; i++) {
        write_manifest_vtk(fields.fields[i]->name, dt, nt, sampling_rate, 1, 0,
                           log_file);
    }

    write_scalar_vtk(vx, 0, 0, log_file);
    write_scalar_vtk(vy, 0, 0, log_file);
    write_scalar_vtk(p, 0, 0, log_file);
    write_scalar_vtk(div, 0, 0, log_file);
    for (int i = 0; i < fields.nb_fields; i++)
        write_scalar_vtk(fields.fields[i], 0, 0, log_file);

    // Main time loop
    // bool inverted = false;
    bool first_loop = true;
    for (unsigned int i = 1; i < nt; i++) {
        log_file << "\n";
        LOG_INFO(log_file, "Starting time loop " << i << " out of " << nt);

        divergence_pic(vx, vy, div, log_file);

        // Making sure that the mean of the divergence is zero
        // Comes from an integral condition to have a solution
        float sum = 0;
        for (unsigned int j = 0; j < ny; j++)
            for (unsigned int i = 0; i < nx; i++)
                sum += GET(div, i, j);

        LOG_INFO(log_file, "Total divergence: " << sum);

        if (data["iteration_algo"] == "Jacobi")
            jacobi_pic(p, temp_p, div, vx, vy, dom, tol, dt, rho, max_iter,
                       first_loop, log_file);
        else if (data["iteration_algo"] == "SOR")
            sor_pic(p, div, vx, vy, dom, tol, dt, rho, max_iter, log_file);
        else {
            LOG_ERR(log_file, "Iteration algorithm not supported");
            return EXIT_FAILURE;
        }

        project_velocity_pic(p, vx, vy, dom, dx, dt, rho, log_file);

        // This is to be able to save it. It serves no purpose in the
        // algorithm
        divergence_pic(vx, vy, div, log_file);

        grid_speed_to_particles(particles, vx, vy, log_file);

        // save files, when the divergence is zero
        if (sampling_rate && !(i % sampling_rate)) {
            write_scalar_vtk(vx, i, 0, log_file);
            write_scalar_vtk(vy, i, 0, log_file);
            write_scalar_vtk(p, i, 0, log_file);
            write_scalar_vtk(div, i, 0, log_file);
            for (int k = 0; k < fields.nb_fields; k++) {
                write_scalar_vtk(fields.fields[k], i, 0, log_file);
            }

            write_particles_vtp(particles, i, 0, 2, log_file);
        }

        advect_pic(particles, vx, vy, dt, log_file);

        particles_speed_to_grid(particles, vx, vy, kern_sum, log_file);

        // advect
        // advect_pic_old(vx, vy, dt, vx, temp_vx, log_file);
        // advect_pic_old(vx, vy, dt, vy, temp_vy, log_file);

        for (int k = 0; k < fields.nb_fields; k++) {
            scalar_field *tmp = scalar_field_copy(fields.fields[k], log_file);
            advect_pic_old(vx, vy, dt, fields.fields[k], tmp, log_file);
            scalar_field *swap = fields.fields[k];
            fields.fields[k] = tmp;
            scalar_field_free(swap, log_file);
        }

        // advect_pic(particles, vx, vy, dt, log_file);

        // inverted = !inverted;
        // scalar_field *invert_vx = vx;
        // scalar_field *invert_vy = vy;

        // vx = temp_vx;
        // vy = temp_vy;

        // temp_vx = invert_vx;
        // temp_vy = invert_vy;

        first_loop = false;

        // Resetting the pressure field
#pragma omp parallel for collapse(2)
        for (unsigned int j = 0; j < ny; j++)
            for (unsigned int i = 0; i < nx; i++)
                SET(p, i, j, 0.0f);
    }

    // As we're not using objects, we need this
    scalar_field_free(vx, log_file);
    scalar_field_free(vy, log_file);
    scalar_field_free(p, log_file);
    scalar_field_free(div, log_file);
    scalar_field_free(dom, log_file);

    // scalar_field_free(temp_vx, log_file);
    // scalar_field_free(temp_vy, log_file);
    scalar_field_free(temp_p, log_file);

    user_field_free(&fields, log_file);

    free(particles);

    auto t1 = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(t1 - t0).count();
    LOG_INFO(log_file, "Total simulation time: " << seconds << " seconds");

    return EXIT_SUCCESS;
}
