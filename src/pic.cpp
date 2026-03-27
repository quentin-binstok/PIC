#include "conditions.hpp"
#include "data.hpp"
#include "nlohmann/json.hpp"
#include "utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
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
    x_1 = std::min(x_1, vx->nx - 2);
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

    float x0 = (x_1 + vx->x_internal) * dx;
    float y0 = (y_1 + vx->y_internal) * dx;

    *v_x =
        interpolate_bilinear(x, y, x0, y0, GET(vx, x_1, y_1), GET(vx, x_2, y_1),
                             GET(vx, x_1, y_2), GET(vx, x_2, y_2), dx, dx);

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
    y_1 = std::min(y_1, vy->ny - 2);
    if (y_1 > vy->ny - 1)
        LOG_WARN(log_file, "y_1 too big")
    y_2 = y_1 + 1;
    y_2 = std::min(vy->ny - 1, y_2);

    x0 = (x_1 + vy->x_internal) * dx;
    y0 = (y_1 + vy->y_internal) * dx;

    *v_y =
        interpolate_bilinear(x, y, x0, y0, GET(vy, x_1, y_1), GET(vy, x_2, y_1),
                             GET(vy, x_1, y_2), GET(vy, x_2, y_2), dx, dx);

    return EXIT_SUCCESS;
}

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
                                   scalar_field *vy, scalar_field *kern_sum_vx,
                                   scalar_field *kern_sum_vy,
                                   std::ofstream &log_file) {
    LOG_INFO(log_file, "Transferring the speed of particles to the grid");

    int nx = vy->nx, ny = vx->ny;
    float dx = vx->dx;

#pragma omp parallel for collapse(2)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            SET(vx, i, j, 0);
            SET(vy, i, j, 0);
            SET(kern_sum_vx, i, j, 0);
            SET(kern_sum_vy, i, j, 0);
        }
    }

    for (int k = 0; k < particles->N; k++) {
        float x = particles->xyz[2 * k];
        float y = particles->xyz[2 * k + 1];
        float u = particles->velocity[2 * k];
        float v = particles->velocity[2 * k + 1];

        // --- vx stencil: staggered in x, collocated in y ---
        int i_vx = std::max(0, (int)(x / dx - 0.5)); // matches get_speed
        int j_vx = std::max(0, (int)(y / dx));
#pragma omp parallel for collapse(2)
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
#pragma omp parallel for collapse(2)
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

inline int grid_speed_to_particles(particle_field *particles, scalar_field *vx,
                                   scalar_field *vy, scalar_field *vx_old,
                                   scalar_field *vy_old, float flip_param,
                                   std::ofstream &log_file) {

#pragma omp parallel for
    for (int k = 0; k < particles->N; k++) {
        float x = particles->xyz[2 * k];
        float y = particles->xyz[2 * k + 1];

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
                          scalar_field *dom, std::vector<float> speed_condition,
                          std::ofstream &log_file) {
    LOG_INFO(log_file, "Computing the divergence");
    int nx = div->nx;
    int ny = div->ny;
    float dx = div->dx;

#pragma omp parallel for collapse(2)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {

            float dudx = 0.0f;
            float dvdy = 0.0f;
            if (i != 0)
                dudx = (GET(vx, i, j) - GET(vx, i - 1, j)) / dx;
            else if (GET(dom, 0, ny / 2) == AIR)
                dudx = 0;
            else
                dudx = (GET(vx, i, j) - speed_condition[0]) / dx;
            if (j != 0)
                dvdy = (GET(vy, i, j) - GET(vy, i, j - 1)) / dx;
            else if (GET(dom, nx / 2, 0) == AIR)
                dvdy = 0;
            else
                dvdy = (GET(vy, i, j) - speed_condition[3]) / dx;

            float d = dudx + dvdy;
            SET(div, i, j, d);
        }
    }
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
    LOG_INFO(log_file, "Computing the residual");

    int nx = p->nx;
    int ny = p->ny;

    float dx = dom->dx;
    const float alpha = dx * dx * rho / dt;
    float beta = rho * dx / dt;

    float norm_squared = 0;

#pragma omp parallel for collapse(2) reduction(+ : norm_squared)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            int cell = GET(dom, i, j);

            if (cell == SOLID)
                continue;

            float p_left, p_right, p_down, p_up;

            // LEFT
            if (i == 0)
                p_left = GET(p, i, j);
            else {
                int l = GET(dom, i - 1, j);
                if (l == AIR) {
                    p_left = 0.f;
                } else if (l == SOLID) {
                    p_left = GET(p, i, j) - beta * GET(vx, i - 1, j);
                } else {
                    p_left = GET(p, i - 1, j);
                }
            }

            // RIGHT
            if (i == nx - 1)
                p_right = GET(p, i, j);
            else {
                int r = GET(dom, i + 1, j);
                if (r == AIR) {
                    p_right = 0.f;
                } else if (r == SOLID) {
                    p_right = GET(p, i, j) + beta * GET(vx, i, j);
                } else {
                    p_right = GET(p, i + 1, j);
                }
            }

            // DOWN
            if (j == 0)
                p_down = GET(p, i, j);
            else {
                int d = GET(dom, i, j - 1);
                if (d == AIR) {
                    p_down = 0.f;
                } else if (d == SOLID) {
                    p_down = GET(p, i, j) - beta * GET(vy, i, j - 1);
                } else {
                    p_down = GET(p, i, j - 1);
                }
            }

            // UP
            if (j == ny - 1)
                p_up = GET(p, i, j);
            else {
                int u = GET(dom, i, j + 1);
                if (u == AIR) {
                    p_up = 0.f;
                } else if (u == SOLID) {
                    p_up = GET(p, i, j) + beta * GET(vy, i, j);
                } else {
                    p_up = GET(p, i, j + 1);
                }
            }

            float new_p =
                (p_left + p_right + p_down + p_up - alpha * GET(div, i, j)) *
                0.25f;

            float r = GET(p, i, j) - new_p;

            norm_squared += r * r;
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

    if (first_loop)
        max_iter = nx * ny;

    float norm_b = 0;

    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
            norm_b += alpha * alpha * GET(div, i, j) * GET(div, i, j);

    norm_b = std::sqrt(norm_b) + 1e-7f;

    float residue = residual_pic(p, vx, vy, div, dom, rho, dt, log_file);
    float condition = residue / norm_b;

    bool inverted = false;
    bool loop = true;

    while (condition > tol && iter < max_iter) {
        if (iter % 100 == 0) {
            LOG_INFO(log_file, "Jacobi on iteration "
                                   << iter << ", criterion is " << condition);
        }
        residue = 0;

#pragma omp parallel for collapse(2) reduction(+ : residue)
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int cell = GET(dom, i, j);

                if (cell == AIR || cell == DIRICHLET) {
                    loop = false;
                }

                if (cell == SOLID) {
                    continue;
                }

                float p_left, p_right, p_down, p_up;

                if (i == 0) {
                    p_left = GET(p, i, j);
                } else {
                    int l = GET(dom, i - 1, j);
                    if (l == AIR) {
                        p_left = 0.f;
                    } else if (l == SOLID) {
                        p_left = GET(p, i, j) - beta * GET(vx, i - 1, j);
                    } else {
                        p_left = GET(p, i - 1, j);
                    }
                }

                if (i == nx - 1) {
                    p_right = GET(p, i, j);
                } else {
                    int r = GET(dom, i + 1, j);
                    if (r == AIR) {
                        p_right = 0.f;
                    } else if (r == SOLID) {
                        p_right = GET(p, i, j) + beta * GET(vx, i, j);
                    } else {
                        p_right = GET(p, i + 1, j);
                    }
                }

                if (j == 0) {
                    p_down = GET(p, i, j);
                } else {
                    int d = GET(dom, i, j - 1);
                    if (d == AIR) {
                        p_down = 0.f;
                    } else if (d == SOLID) {
                        p_down = GET(p, i, j) - beta * GET(vy, i, j - 1);
                    } else {
                        p_down = GET(p, i, j - 1);
                    }
                }

                if (j == ny - 1) {
                    p_up = GET(p, i, j);
                } else {
                    int u = GET(dom, i, j + 1);
                    if (u == AIR) {
                        p_up = 0.f;
                    } else if (u == SOLID) {
                        p_up = GET(p, i, j) + beta * GET(vy, i, j);
                    } else {
                        p_up = GET(p, i, j + 1);
                    }
                }

                float new_p = (p_left + p_right + p_down + p_up -
                               alpha * GET(div, i, j)) *
                              0.25f;

                float r = GET(p, i, j) - new_p;
                residue += r * r;

                SET(temp_p, i, j, new_p);
            }
        }

        if (loop) {
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
        }

        std::swap(p, temp_p);
        inverted = !inverted;

        residue = std::sqrt(residue);
        condition = residue / norm_b;

        iter++;
    }

    if (iter == max_iter) {
        LOG_WARN(log_file, "Jacobi stopped at " << max_iter << " iterations");
    } else {
        LOG_INFO(log_file, "Jacobi converged in " << iter << " iterations");
    }
    LOG_INFO(log_file, "Last residue: " << residue);

    if (inverted)
        std::swap(p, temp_p);

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
    bool loop = true;

    while ((condition > tol) && iter < max_iter) {
        if (iter % 100 == 0) {
            LOG_INFO(log_file, "SOR on iteration " << iter << ", criterion is "
                                                   << condition);
        }
        residue = 0;
        // to be able to parallelize, need checkered grids
        for (int color = 0; color < 2; color++) {

#pragma omp parallel for collapse(2) reduction(+ : residue)
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    if ((i + j) % 2 != color)
                        continue;

                    int cell = GET(dom, i, j);

                    if (cell == AIR || cell == DIRICHLET) {
                        loop = false;
                    }

                    if (cell == SOLID) {
                        continue;
                    }

                    if (cell == AIR) {
                        SET(p, i, j, 0);
                        continue;
                    }

                    float p_left = 0, p_right = 0, p_down = 0, p_up = 0;

                    if (i == 0) {
                        p_left = GET(p, i, j);
                    } else {
                        int l = GET(dom, i - 1, j);
                        if (l == SOLID) {
                            p_left = GET(p, i, j) - beta * GET(vx, i - 1, j);
                        } else if (l == AIR) {
                            p_left = 0;
                        } else {
                            p_left = GET(p, i - 1, j);
                        }
                    }

                    if (i == nx - 1) {
                        p_right = GET(p, i, j);
                    } else {
                        int r = GET(dom, i + 1, j);
                        if (r == SOLID) {
                            p_right = GET(p, i, j) + beta * GET(vx, i, j);
                        } else if (r == AIR) {
                            p_right = 0;
                        } else {
                            p_right = GET(p, i + 1, j);
                        }
                    }

                    if (j == 0) {
                        p_down = GET(p, i, j);
                    } else {
                        int d = GET(dom, i, j - 1);
                        if (d == SOLID) {
                            p_down = GET(p, i, j) - beta * GET(vy, i, j - 1);
                        } else if (d == AIR) {
                            p_down = 0;
                        } else {
                            p_down = GET(p, i, j - 1);
                        }
                    }

                    if (j == ny - 1) {
                        p_up = GET(p, i, j);
                    } else {
                        int u = GET(dom, i, j + 1);
                        if (u == SOLID) {
                            p_up = GET(p, i, j) + beta * GET(vy, i, j);
                        } else if (u == AIR) {
                            p_up = 0;
                        } else {
                            p_up = GET(p, i, j + 1);
                        }
                    }

                    float new_p = (p_left + p_right + p_down + p_up -
                                   alpha * GET(div, i, j)) /
                                  4.0;
                    residue += (GET(p, i, j) - new_p) * (GET(p, i, j) - new_p);
                    SET(p, i, j, GET(p, i, j) + omega * (new_p - GET(p, i, j)));
                }
            }
        }
        if (loop) {
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
        }

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
                                float dt, float rho, std::ofstream &log_file,
                                std::vector<float> &speed_condition) {
    LOG_INFO(log_file, "Projecting the velocity field")
    int vx_nx = vx->nx;
    int vx_ny = vx->ny;
    int vy_nx = vy->nx;
    int vy_ny = vy->ny;
    int smooth = -1;
#pragma omp parallel for collapse(2)
    for (int j = 0; j < vx_ny; j++) {
        for (int i = 0; i < vx_nx; i++) {
            if (GET(dom, i, j) == SOLID || GET(dom, i + 1, j) == SOLID) {
                SET(vx, i, j, 0.0);
                continue;
            }
            if (GET(dom, i, j) == DIRICHLET) {
                SET(vx, i, j, 0);
                if (i == 0) {
                    float v = speed_condition[0];
                    if (j < smooth)
                        v *= (float)(j) / smooth;
                    if (j > vx_ny - smooth)
                        v *= (float)(vx_ny - j - 1) / smooth;
                    SET(vx, i, j, v);
                }

                if (i == vx_nx - 1) {
                    float v = speed_condition[1];
                    if (j < smooth)
                        v *= (float)j / smooth;
                    if (j > vx_ny - smooth)
                        v *= (float)(vx_ny - j - 1) / smooth;
                    SET(vx, i, j, v);
                }

                continue;
            }
            if (i == vx_nx - 1) {
                float gradp_x = (GET(p, i, j) - GET(p, i - 1, j)) / dx;
                SET(vx, i, j, GET(vx, i, j) - dt * gradp_x / rho);
                continue;
            }
            float gradp_x = (GET(p, i + 1, j) - GET(p, i, j)) / dx;
            SET(vx, i, j, GET(vx, i, j) - dt * gradp_x / rho);
        }
    }

#pragma omp parallel for collapse(2)
    for (int j = 0; j < vy_ny; j++) {
        for (int i = 0; i < vy_nx; i++) {
            if (GET(dom, i, j) == SOLID || GET(dom, i, j + 1) == SOLID) {
                SET(vy, i, j, 0.0);
                continue;
            }
            if (GET(dom, i, j) == DIRICHLET) {
                SET(vy, i, j, 0);
                if (j == 0) {
                    float v = speed_condition[3];
                    if (i < smooth)
                        v *= (float)i / smooth;
                    if (i > vy_nx - smooth)
                        v *= (float)(vy_nx - i - 1) / smooth;
                    SET(vy, i, j, v);
                }

                if (j == vy_ny - 1) {
                    float v = speed_condition[2];
                    if (i < smooth)
                        v *= (float)i / smooth;
                    if (i > vy_nx - smooth)
                        v *= (float)(vy_nx - i - 1) / smooth;
                    SET(vy, i, j, v);
                }
                continue;
            }
            if (j == vy_ny - 1) {
                float gradp_y = (GET(p, i, j) - GET(p, i, j - 1)) / dx;
                SET(vy, i, j, GET(vy, i, j) - dt * gradp_y / rho);
                continue;
            }
            float gradp_y = (GET(p, i, j + 1) - GET(p, i, j)) / dx;
            SET(vy, i, j, GET(vy, i, j) - dt * gradp_y / rho);
        }
    }

    // After the P2G normalization loop, add:

    // Free-slip on left/right walls: ∂vy/∂x = 0
    for (int j = 0; j < vy_ny; j++) {
        SET(vy, 0, j, GET(vy, 1, j));                 // left wall
        SET(vy, vy_nx - 1, j, GET(vy, vy_nx - 2, j)); // right wall
    }

    // Symmetrically, for vx on horizontal walls: ∂vx/∂y = 0
    for (int i = 0; i < vx_nx; i++) {
        SET(vx, i, 0, GET(vx, i, 1));                 // bottom wall
        SET(vx, i, vx_ny - 1, GET(vx, i, vx_ny - 2)); // top wall
    }

    return EXIT_SUCCESS;
}

void apply_gravity(particle_field *particles, float g, float dt) {
    //     int nx = dom->nx, ny = dom->ny;

    // #pragma omp parallel for collapse(2)
    //     for (int j = 0; j < ny; j++) {
    //         for (int i = 0; i < nx; i++) {
    //             float cell_type = GET(dom, i, j);
    //             float cell_upper;
    //             if (j != ny - 1)
    //                 cell_upper = GET(dom, i, j + 1);
    //             else
    //                 cell_upper = SOLID;
    //             if (cell_type == LIQUID || cell_upper == LIQUID) {
    //                 float init_vy = GET(vy, i, j);
    //                 SET(vy, i, j, init_vy - g * dt);
    //             }
    //         }
    //     }

#pragma omp parallel for
    for (int k = 0; k < particles->N; k++) {
        particles->velocity[2 * k + 1] -= g * dt;
    }
}

inline void initialize_particles_pic(particle_field *particles,
                                     scalar_field *dom, scalar_field *vx,
                                     scalar_field *vy, int density,
                                     std::ofstream &log_file) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-0.5 * dom->dx, 0.5 * dom->dx);
    int nx = dom->nx, ny = dom->ny;
    float dx = dom->dx;

    int nb_not_fluid_cases = 0;
    int current_id = 0;

    particles->xyz.resize(2 * density * nx * ny);
    particles->velocity.resize(2 * density * nx * ny);

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            float cell = GET(dom, i, j);
            if (cell != SOLID && cell != AIR) {
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

    // After the loop:
    particles->N = current_id;
    particles->next_id = current_id; // next to assign = current_id
    particles->xyz.resize(2 * current_id);
    particles->velocity.resize(2 * current_id);

    // Fill ids properly
    particles->id.resize(current_id);
    std::iota(particles->id.begin(), particles->id.end(), 0);
}

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

inline int update_particles_pic(particle_field *particles, scalar_field *vx,
                                scalar_field *vy, scalar_field *dom,
                                int creation_rate, float dt,
                                std::vector<int> &density,
                                std::ofstream &log_file) {
    LOG_INFO(log_file, "Updating particles")

    int nx = dom->nx;
    int ny = dom->ny;
    float dx = dom->dx;

    std::fill(density.begin(), density.end(), 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> jitter(-0.5 * dx, 0.5 * dx);
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    std::uniform_real_distribution<float> birth_dist(0.0f, dt);

    // Remove invalid particles
    int p = 0;
    while (p < particles->N) {

        float x = particles->xyz[2 * p];
        float y = particles->xyz[2 * p + 1];

        int i = (int)floor(x / dx + 0.5f);
        int j = (int)floor(y / dx + 0.5f);

        // Check if the particle is out of bounds
        if (i < 0 || j < 0 || i >= nx || j >= ny) {
            remove_particle(particles, p);
            continue;
        }

        // Check if the particle is in a solid cell
        float cell = GET(dom, i, j);
        if (cell == SOLID) {
            // remove_particle(particles, p);
            // particles->velocity[2 * p] *= -1;
            // particles->velocity[2 * p + 1] *= -1;

            if (i == 0) {
                // particles->velocity[2 * p] *= -1;
                particles->xyz[2 * p] = 2 * dx / 3;
            } else if (i == nx - 1) {
                particles->xyz[2 * p] = (nx-1) * dx - 2 * dx / 3;
            } else if (j == 0) {
                particles->xyz[2 * p + 1] = 2 * dx / 3;
            } else if (j == ny - 1) {
                particles->xyz[2 * p + 1] = (ny-1) * dx - 2 * dx / 3;
            }
            // } else if (j == 0 || j == ny - 1) {
            //     particles->velocity[2 * p + 1] *= -1;
            else {
                remove_particle(particles, p);
                continue;
            }
            // continue;
        }

        density[j * nx + i]++;
        p++;
    }

    // Emit new particles from (inflow) cells
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {

            float cell = GET(dom, i, j);

            if (cell == DIRICHLET) {
                float n = dt * creation_rate;
                // float frac = n - floor(n);
                int to_add = (int)floor(n); // + (prob(gen) < frac ? 1 : 0);

                for (int k = 0; k < to_add; k++) {

                    // Jittered spawn position within the cell
                    float x = i * dx + jitter(gen);
                    float y = j * dx + jitter(gen);

                    float vx_p, vy_p;
                    get_speed(&vx_p, &vy_p, x, y, vx, vy, log_file);

                    float tau = birth_dist(gen);
                    float remaining = dt - tau;
                    advect_single_particle(&x, &y, vx, vy, remaining, log_file);

                    int fi = (int)floor(x / dx + 0.5f);
                    int fj = (int)floor(y / dx + 0.5f);

                    if (fi < 0 || fj < 0 || fi >= nx || fj >= ny) {
                        continue;
                    } else if (GET(dom, fi, fj) == SOLID) {
                        continue;
                    } else {
                        particles->xyz.push_back(x);
                        particles->xyz.push_back(y);

                        particles->velocity.push_back(vx_p);
                        particles->velocity.push_back(vy_p);

                        particles->id.push_back(particles->next_id);
                        particles->next_id++;
                        particles->N++;

                        density[fj * nx + fi]++;
                    }
                }
            }
        }
    }

    return EXIT_SUCCESS;
}

void fill_cell(int i, int j, particle_field *particles, scalar_field *vx,
               scalar_field *vy, scalar_field *dom, int imposed_density,
               std::vector<int> density, float dt, std::ofstream &log_file) {
    // LOG_INFO(log_file, "Filling cell " << i << " " << j);
    int nx = vx->nx, ny = vx->ny;
    float dx = vx->dx;

    int cell_density = density[j * nx + i];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> jitter(-0.5 * dx, 0.5 * dx);
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    std::uniform_real_distribution<float> birth_dist(0.0f, dt);

    int to_add = std::max(0, (int)floor(imposed_density - cell_density));

    for (int k = 0; k < to_add; k++) {
        // Jittered spawn position within the cell
        float x = i * dx + jitter(gen);
        float y = j * dx + jitter(gen);

        float vx_p, vy_p;
        get_speed(&vx_p, &vy_p, x, y, vx, vy, log_file);

        float tau = birth_dist(gen);
        float remaining = dt - tau;
        advect_single_particle(&x, &y, vx, vy, remaining, log_file);

        int fi = (int)floor(x / dx + 0.5f);
        int fj = (int)floor(y / dx + 0.5f);

        if (fi < 0 || fj < 0 || fi >= nx || fj >= ny) {
            continue;
        } else if (GET(dom, fi, fj) == SOLID) {
            continue;
        } else {
            particles->xyz.push_back(x);
            particles->xyz.push_back(y);

            particles->velocity.push_back(vx_p);
            particles->velocity.push_back(vy_p);

            particles->id.push_back(particles->next_id);
            particles->next_id++;
            particles->N++;

            density[fj * nx + fi]++;
        }
    }
    // LOG_INFO(log_file, "Ended filling cell");
}

void refill_domain(particle_field *particles, scalar_field *dom,
                   scalar_field *vx, scalar_field *vy,
                   std::vector<int> &density, int particle_density,
                   float percent_limit, float dt, std::ofstream &log_file) {
    LOG_INFO(log_file, "Refilling domain");
    int nx = dom->nx, ny = dom->ny;
    float dx = dom->dx;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> jitter(-0.5 * dx, 0.5 * dx);
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    std::uniform_real_distribution<float> birth_dist(0.0f, dt);

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            float cell_type = GET(dom, i, j);
            int cell_density = density[j * nx + i];
            bool solid_neighbour = false;
            if (GET(dom, i + 1, j) == SOLID || GET(dom, i - 1, j) == SOLID ||
                GET(dom, i, j + 1) == SOLID || GET(dom, i, j - 1) == SOLID)
                solid_neighbour = true;

            if (cell_type == LIQUID) {
                if (cell_density >= percent_limit * particle_density &&
                    !solid_neighbour) {
                    fill_cell(i, j, particles, vx, vy, dom, particle_density,
                              density, dt, log_file);
                }
                // change to air
                else if (cell_density < 1) {
                    SET(dom, i, j, AIR);
                }
            } else if (cell_type == AIR && i != 0 && i != nx - 1 && j != 0 &&
                       j != ny - 1) {
                if (cell_density > 0) {
                    SET(dom, i, j, LIQUID);
                }
            }
            // float up_cell = GET(dom, i, j + 1);
            // float down_cell = GET(dom, i, j - 1);
            // float right_cell = GET(dom, i + 1, j);
            // float left_cell = GET(dom, i - 1, j);
            // if (up_cell == LIQUID && down_cell == LIQUID &&
            //     right_cell == LIQUID && left_cell == LIQUID) {
            //     SET(dom, i, j, LIQUID);

            //     fill_cell(i, j, particles, vx, vy, dom, particle_density,
            //               density, dt, log_file);
            // }
        }
    }
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

    int particle_density = data.value("particle_density", 8);
    float speed_x = 0.0f;
    for (const auto &bc : data["bc"]) {
        if (bc.contains("type") && bc["type"].get<float>() == 3.0f) {
            if (bc.contains("speed_x")) {
                speed_x = bc["speed_x"].get<float>();
            }
        }
    }
    int creation_rate = particle_density * speed_x  / dx;
    float percent_limit = data.value("particle_percentage_limit", 0.3);
    float flip_param = data.value("flip", 0.0f);
    LOG_INFO(log_file, "FLIP percentage is " << flip_param * 100);

    bool gravity = data.value("gravity", false);
    float g = data.value("g", 9.81);

    user_fields fields;
    get_fields(&fields, data, log_file);

    // Initialising the fields
    scalar_field *vx = scalar_field_init("vx", nx, ny, 0.5, 0, dx, log_file);
    scalar_field *vy = scalar_field_init("vy", nx, ny, 0, 0.5, dx, log_file);
    scalar_field *p = scalar_field_init("p", nx, ny, 0, 0, dx, log_file);
    scalar_field *div = scalar_field_init("div", nx, ny, 0, 0, dx, log_file);
    scalar_field *dom = scalar_field_init("dom", nx, ny, 0, 0, dx, log_file);

    scalar_field *kern_sum_vx =
        scalar_field_init("kern_sum_vx", nx, ny, 0, 0, dx, log_file);
    scalar_field *kern_sum_vy =
        scalar_field_init("kern_sum_vy", nx, ny, 0, 0, dx, log_file);

    if (!vx || !vy || !p || !div || !dom) {
        LOG_ERR(log_file, "An error occured initializing fields.");
        return EXIT_FAILURE;
    }

    std::vector<float> speed_condition;

    // Applying the initial conditions

    create_circle(dom, "ic_cylinders", data, log_file);
    initialize_speed(vx, dom, data, "ic_vx", log_file);
    initialize_speed(vy, dom, data, "ic_vy", log_file);
    boundary_condition(vx, vy, dom, speed_condition, data, "bc", log_file);
    initialize_domain(dom, data, "ic_cell", log_file);

    scalar_field *temp_vx = scalar_field_copy(vx, log_file);
    scalar_field *temp_vy = scalar_field_copy(vy, log_file);
    scalar_field *temp_p = scalar_field_copy(p, log_file);

    // Initializing the particles
    particle_field *particles = particle_field_init_2D(
        "particles", nx * ny * particle_density, log_file);
    initialize_particles_pic(particles, dom, vx, vy, data["particle_density"],
                             log_file);

    write_manifest_vtk("particles", dt, nt, sampling_rate, 1, 1, log_file);
    write_particles_vtp(particles, 0, 0, 2, log_file);

    write_manifest_vtk(vx->name, dt, nt, sampling_rate, 1, 0, log_file);
    write_manifest_vtk(vy->name, dt, nt, sampling_rate, 1, 0, log_file);
    write_manifest_vtk(p->name, dt, nt, sampling_rate, 1, 0, log_file);
    write_manifest_vtk(div->name, dt, nt, sampling_rate, 1, 0, log_file);
    write_manifest_vtk(dom->name, dt, nt, sampling_rate, 1, 0, log_file);
    for (int i = 0; i < fields.nb_fields; i++) {
        write_manifest_vtk(fields.fields[i]->name, dt, nt, sampling_rate, 1, 0,
                           log_file);
    }

    write_scalar_vtk(vx, 0, 0, log_file);
    write_scalar_vtk(vy, 0, 0, log_file);
    write_scalar_vtk(p, 0, 0, log_file);
    write_scalar_vtk(div, 0, 0, log_file);
    write_scalar_vtk(dom, 0, 0, log_file);
    for (int i = 0; i < fields.nb_fields; i++)
        write_scalar_vtk(fields.fields[i], 0, 0, log_file);

    std::vector<int> density(nx * ny, 0);

    // Main time loop
    // bool inverted = false;
    bool first_loop = true;
    for (unsigned int i = 1; i < nt; i++) {
        log_file << "\n";
        LOG_INFO(log_file, "Starting time loop " << i << " out of " << nt);

        if (gravity)
            apply_gravity(particles, g, dt);

        particles_speed_to_grid(particles, vx, vy, kern_sum_vx, kern_sum_vy,
                                log_file);

        divergence_pic(vx, vy, div, dom, speed_condition, log_file);

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

        std::memcpy(temp_vx->values, vx->values, nx * ny * sizeof(float));
        std::memcpy(temp_vy->values, vy->values, nx * ny * sizeof(float));

        project_velocity_pic(p, vx, vy, dom, dx, dt, rho, log_file,
                             speed_condition);

        // This is to be able to save it. It serves no purpose in the
        // algorithm
        divergence_pic(vx, vy, div, dom, speed_condition, log_file);

        grid_speed_to_particles(particles, vx, vy, temp_vx, temp_vy, flip_param,
                                log_file);

        // save files, when the divergence is zero
        if (sampling_rate && !(i % sampling_rate)) {
            write_scalar_vtk(vx, i, 0, log_file);
            write_scalar_vtk(vy, i, 0, log_file);
            write_scalar_vtk(p, i, 0, log_file);
            write_scalar_vtk(div, i, 0, log_file);
            write_scalar_vtk(dom, i, 0, log_file);
            for (int k = 0; k < fields.nb_fields; k++) {
                write_scalar_vtk(fields.fields[k], i, 0, log_file);
            }

            write_particles_vtp(particles, i, 0, 2, log_file);
        }

        advect_pic(particles, vx, vy, dt, log_file);

        std::fill(density.begin(), density.end(), 0);
        update_particles_pic(particles, vx, vy, dom, creation_rate, dt, density,
                             log_file);
        refill_domain(particles, dom, vx, vy, density, particle_density,
                      percent_limit, dt, log_file);

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
    }

    // As we're not using objects, we need this
    scalar_field_free(vx, log_file);
    scalar_field_free(vy, log_file);
    scalar_field_free(p, log_file);
    scalar_field_free(div, log_file);
    scalar_field_free(dom, log_file);

    scalar_field_free(kern_sum_vx, log_file);
    scalar_field_free(kern_sum_vy, log_file);

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
