#include "poisson.hpp"
#include "conditions.hpp"
#include "data.hpp"
#include "nlohmann/json.hpp"
#include "utils.hpp"

/*
 @brief computes the residual term of the pressure computation
 @params scarlar_field res: where the resulting Ax term will be stored
 @params same as everywhere
 @returns: the 2-norm of the residual
*/
float residual(scalar_field *p, scalar_field *vx, scalar_field *vy,
               scalar_field *div, scalar_field *dom, float rho, float dt,
               std::ofstream &log_file) {
    LOG_INFO(log_file, "Computing the residual");

    int nx = p->nx;
    int ny = p->ny;

    float dx = dom->dx;
    float alpha = dx * dx * rho / dt;
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
int jacobi(scalar_field *p, scalar_field *temp_p, scalar_field *div,
           scalar_field *vx, scalar_field *vy, scalar_field *dom, float tol,
           float dt, float rho, int max_iter, bool first_loop,
           std::ofstream &log_file) {
    LOG_INFO(log_file, "Starting Jacobi");

    int nx = p->nx;
    int ny = p->ny;

    float dx = dom->dx;
    float alpha = dx * dx * rho / dt;
    float beta = rho * dx / dt;

    int iter = 0;

    if (first_loop)
        max_iter = nx * ny;

    float norm_b = 0;

    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
            norm_b += alpha * alpha * GET(div, i, j) * GET(div, i, j);

    norm_b = std::sqrt(norm_b) + 1e-7f;

    float residue = residual(p, vx, vy, div, dom, rho, dt, log_file);
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

                if (cell == AIR) {
                    SET(p, i, j, 0);
                    continue;
                }

                float p_left = 0, p_right = 0, p_down = 0, p_up = 0;

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
int sor(scalar_field *p, scalar_field *div, scalar_field *vx, scalar_field *vy,
        scalar_field *dom, float tol, float dt, float rho, int max_iter,
        std::ofstream &log_file) {
    LOG_INFO(log_file, "Starting SOR")

    int nx = p->nx;
    int ny = p->ny;
    float dx = dom->dx;
    float alpha = dx * dx * rho / dt;
    float beta = rho * dx / dt;
    int iter = 0;

    // parameters needed for the algorithm
    int N = std::min(nx, ny);
    float pi = 3.14159265358979;
    float omega = std::min(1.95f, 2.0f / (1.0f + std::sin(pi / N)));

    float norm_b = 0;
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            norm_b += alpha * alpha * GET(div, i, j) * GET(div, i, j);
        }
    }
    norm_b = std::sqrt(norm_b);
    norm_b += 1e-7;

    float residue = residual(p, vx, vy, div, dom, rho, dt, log_file);
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