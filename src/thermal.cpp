#include "thermal.hpp"
#include "data.hpp"
#include "nlohmann/json.hpp"
#include "particules.hpp"
#include "utils.hpp"
#include <limits>

using json = nlohmann::json;

/*
 @brief brings the particule fields to the grid
*/
int particles_temp_to_grid(particle_field *particles, scalar_field *T,
                           std::vector<bool> &changed, scalar_field *kern_sum_T,
                           std::ofstream &log_file) {
    LOG_INFO(log_file, "Transferring the temperature of particles to the grid");

    int nx = T->nx, ny = T->ny;
    float dx = T->dx;

    // Buffer, because can't zero out T
    std::vector<float> T_accum(nx * ny, 0.0f);

// Zero things out
#pragma omp parallel for collapse(2)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            SET(kern_sum_T, i, j, 0);
        }
    }

    // Gets the speeds, the kernel weights
#pragma omp parallel for
    for (int k = 0; k < particles->N; k++) {
        float x = particles->xyz[2 * k];
        float y = particles->xyz[2 * k + 1];
        float temp = particles->T[k];

        int i_t = std::max(0, (int)(x / dx)); // matches get_speed
        int j_t = std::max(0, (int)(y / dx));

        for (int j = std::max(j_t - 2, 0); j < std::min(ny, j_t + 2); j++) {
            for (int i = std::max(i_t - 2, 0); i < std::min(nx, i_t + 2); i++) {
                float kern =
                    kernel((x - i * dx) / dx) * kernel((y - j * dx) / dx);

#pragma omp atomic
                T_accum[j * nx + i] += temp * kern;

#pragma omp atomic
                kern_sum_T->values[j * nx + i] += kern;
            }
        }
    }

// Divide by the total kernel weight
#pragma omp parallel for collapse(2)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            float kern = GET(kern_sum_T, i, j);
            // Only replace if particles carried information there
            if (kern > 0) {
                SET(T, i, j, T_accum[j * nx + i] / kern);
            }
        }
    }

    return EXIT_SUCCESS;
}

/*
 @brief interpolates the grid fields to the particles
*/
int grid_temp_to_particles(particle_field *particles, scalar_field *T,
                           std::ofstream &log_file) {
    LOG_INFO(log_file,
             "Transferring the temperature from the grid to particles");

    float dx = T->dx;

#pragma omp parallel for
    for (int k = 0; k < particles->N; k++) {
        float x = particles->xyz[2 * k];
        float y = particles->xyz[2 * k + 1];

        // gets the speeds on the old and new grids
        float temp = interp_temp(x, y, dx, T);

        particles->T[k] = temp;
    }

    return EXIT_SUCCESS;
}

/*
 * Does a SOR to solve the diffusion equation for temperature
 * Implicit solver
 */
void apply_thermal_eq(scalar_field *T, scalar_field *T_temp, therm_bc *bcs,
                      float dt, float c, float rho, float k, float tol,
                      int max_iter, std::ofstream &log_file) {
    LOG_INFO(log_file, "Applying temperature formula");

    // Future-proof, will be the thermal generation term
    float gamma = 0.0f;

    int nx = T->nx, ny = T->ny;
    float dx = T->dx;
    float alpha = (dt * k) / (dx * dx * rho * c);
    LOG_INFO(log_file, "\talpha = " << alpha);

    int iter = 0;
    int N = std::min(nx, ny);
    float pi = 3.14159265358979;
    float omega = std::min(1.95f, 2.0f / (1.0f + std::sin(pi / N)));

    float norm_b = 0;
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            norm_b += (GET(T, i, j) + gamma) * (GET(T, i, j) + gamma);
        }
    }
    norm_b = std::sqrt(norm_b);
    norm_b += 1e-7;

    float residue = 0.0f;
    float condition = std::numeric_limits<float>::max();

    do {
        if (iter % 100 == 0) {
            LOG_INFO(log_file, "Thermal SOR on iteration "
                                   << iter << ", criterion is " << condition);
        }

        residue = 0.0f;

        for (int color = 0; color < 2; color++) {
#pragma omp parallel for collapse(2) reduction(+ : residue)
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    if ((i + j) % 2 != color)
                        continue;

                    float Tn_ij = GET(T, i, j);
                    float Tn1_ij = GET(T_temp, i, j);

                    float top_temp;
                    if (j != ny - 1)
                        top_temp = GET(T_temp, i, j + 1);
                    else {
                        if (bcs->type[2] == DIRICHLET_THERM)
                            top_temp = bcs->val[2];
                        else
                            top_temp = (dx * bcs->val[2]) / k + Tn1_ij;
                    }

                    float bottom_temp;
                    if (j != 0)
                        bottom_temp = GET(T_temp, i, j - 1);
                    else {
                        if (bcs->type[3] == DIRICHLET_THERM) {
                            bottom_temp = bcs->val[3];
                        } else
                            bottom_temp = (dx * bcs->val[3]) / k + Tn1_ij;
                    }

                    float left_temp;
                    if (i != 0)
                        left_temp = GET(T_temp, i - 1, j);
                    else {
                        if (bcs->type[0] == DIRICHLET_THERM)
                            left_temp = bcs->val[0];
                        else
                            left_temp = (dx * bcs->val[0]) / k + Tn1_ij;
                    }

                    float right_temp;
                    if (i != nx - 1)
                        right_temp = GET(T_temp, i + 1, j);
                    else {
                        if (bcs->type[1] == DIRICHLET_THERM)
                            right_temp = bcs->val[1];
                        else
                            right_temp = (dx * bcs->val[1]) / k + Tn1_ij;
                    }

                    float new_temp = (Tn_ij + gamma +
                                      alpha * (top_temp + bottom_temp +
                                               left_temp + right_temp)) /
                                     (1.0f + 4 * alpha);

                    residue += (Tn1_ij - new_temp) * (Tn1_ij - new_temp);
                    SET(T_temp, i, j, Tn1_ij + omega * (new_temp - Tn1_ij));
                }
            }
        }
        residue = std::sqrt(residue);
        condition = residue / norm_b;
        condition = std::abs(condition);
        iter++;
    } while ((condition > tol) && (iter < max_iter));

    float *temp = T->values;
    T->values = T_temp->values;
    T_temp->values = temp;
    LOG_INFO(log_file, "Exiting thermal SOR, converged in " << iter);
}

/*
 * Builds the structure holding the information on the thermal BCs
 */
void build_thermal_bc(therm_bc *bcs, json &data, std::ofstream &log_file) {
    LOG_INFO(log_file, "Applying " << "thermal bcs" << " on the boundaries");

    bcs->type.resize(4);
    bcs->val.resize(4);
    // Easy access to the condition
    auto condition = data["thermal_bcs"];
    for (int k = 0; k < (int)condition.size(); k++) {
        const int type = condition[k]["type"];
        const int side = condition[k]["side"];
        const float val = condition[k]["val"];

        switch (side) {
        case 0: {
            // left boundary
            bcs->type[0] = (THERM_BC_TYPE)type;
            bcs->val[0] = val;
        } break;
        case 1: {
            // right boundary
            bcs->type[1] = (THERM_BC_TYPE)type;
            bcs->val[1] = val;
        } break;
        case 2: {
            // top boundary
            bcs->type[2] = (THERM_BC_TYPE)type;
            bcs->val[2] = val;
        } break;
        case 3: {
            // bottom boundary
            bcs->type[3] = (THERM_BC_TYPE)type;
            bcs->val[3] = val;
        } break;
        default:
            break;
        }
    }

    log_file.flush();

    for (int i = 0; i < 4; i++) {
        LOG_INFO(log_file, "Side: " << i);
        LOG_INFO(log_file, "\tType: " << bcs->type[i]);
        LOG_INFO(log_file, "\tValue: " << bcs->val[i]);
    }
    log_file.flush();
}

/*
 * Small convenience function to interpolate the temperature field
 */
float interp_temp(float x, float y, float dx, scalar_field *T) {
    float temp;
    int x_1, x_2, y_1, y_2;
    x_1 = (int)(x / dx);
    x_1 = std::max(0, x_1);
    x_1 = std::min(x_1, T->nx - 2);
    x_2 = std::min(T->nx - 1, x_1 + 1);

    y_1 = std::max(0, (int)(y / dx));
    y_1 = std::min(y_1, T->ny - 2);
    y_2 = std::min(y_1 + 1, T->ny - 1);

    float x0 = x_1 * dx;
    float y0 = y_1 * dx;
    temp =
        interpolate_bilinear(x, y, x0, y0, GET(T, x_1, y_1), GET(T, x_2, y_1),
                             GET(T, x_1, y_2), GET(T, x_2, y_2), dx, dx);

    return temp;
}