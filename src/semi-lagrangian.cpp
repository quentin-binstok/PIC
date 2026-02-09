#include "conditions.hpp"
#include "data.hpp"
#include "nlohmann/json.hpp"
#include "utils.hpp"
#include <cstdlib>
#include <cstring>
#include <fstream>

using json = nlohmann::json;

/*
 @brief checks the validity of parameters (except boundary and initial
 conditions)
 @param data: the whole data json
 @param log_file: the log file
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
    unsigned int nx = q_n->nx, ny = q_n->ny;
    float x_int = q_n->x_internal, y_int = q_n->y_internal;
    float dx = q_n->dx;

    if (0)
        LOG_INFO(log_file, "test")

    // #pragma omp parallel for collapse(2)
    for (unsigned int j = 0; j < ny; j++) {
        for (unsigned int i = 0; i < nx; i++) {
            // Interpolation of the speed field

            // coords where we need the speed field
            float x = (i + x_int) * dx, y = (j + y_int) * dx;

            // vx
            float v_x = 0;
            int x_1, x_2, y_1, y_2;
            if (y_int > 0) {
                x_1 = (i - 1);
                x_2 = (i);
                y_1 = (j);
                y_2 = (j + 1);
            } else {
                x_1 = (i - 1);
                x_2 = (i);
                y_1 = (j - 1);
                y_2 = (j);
            }

            if (!(x_1 < 0 || y_1 < 0 || x_2 >= (int)vx->nx ||
                  y_2 >= (int)vx->ny))
                v_x = interpolate_bilinear(x, y, (x_1 + vx->x_internal) * dx,
                                           (y_1 + vx->y_internal) * dx,
                                           GET(vx, x_1, y_1), GET(vx, x_2, y_1),
                                           GET(vx, x_1, y_2), GET(vx, x_2, y_2),
                                           dx, dx);
            else {
                LOG_DEBUG(log_file, "edge case")
                // 8 possibilities

                // left side
                if (x_1 < 0 && y_1 >= 0 && y_2 < (int)vx->ny) {
                    LOG_DEBUG(log_file, "left side")
                    v_x = GET(vx, 0, y_1) * (1 - (y - y_1 * dx) / (dx)) +
                          GET(vx, 0, y_2) * (y - y_1 * dx) / dx;
                }

                // right side
                if (x_2 >= (int)vx->nx && y_1 >= 0 && y_2 < (int)vx->ny) {
                    LOG_DEBUG(log_file, "right side")
                    v_x =
                        GET(vx, vx->nx - 1, y_1) * (1 - (y - y_1 * dx) / (dx)) +
                        GET(vx, vx->nx - 1, y_2) * (y - y_1 * dx) / dx;
                }

                // bottom side
                if (y_1 < 0 && x_1 >= 0 && x_2 < (int)vx->nx) {
                    LOG_DEBUG(log_file, "bottom side");
                    v_x = GET(vx, x_1, 0) * (1 - (x - x_1 * dx) / (dx)) +
                          GET(vx, x_2, 0) * (x - x_1 * dx) / dx;
                }

                // top side
                if (y_2 >= (int)vx->ny && x_1 >= 0 && x_2 < (int)vx->nx) {
                    LOG_DEBUG(log_file, "top side");
                    v_x =
                        GET(vx, x_1, vx->ny - 1) * (1 - (x - x_1 * dx) / (dx)) +
                        GET(vx, x_2, vx->ny - 1) * (x - x_1 * dx) / dx;
                }

                // bottom left
                if (x_1 < 0 && y_1 < 0) {
                    LOG_DEBUG(log_file, "botttom left");
                    v_x = GET(vx, 0, 0);
                }

                // top left
                if (x_1 < 0 && y_2 >= (int)vx->ny) {
                    LOG_DEBUG(log_file, "top left");
                    v_x = GET(vx, 0, vx->ny - 1);
                }

                // bottom right
                if (x_2 >= (int)vx->nx && y_1 < 0) {
                    LOG_DEBUG(log_file, "bottom right");
                    v_x = GET(vx, vx->nx - 1, 0);
                }

                // top right
                if (x_2 >= (int)vx->nx && y_2 >= (int)vx->ny) {
                    LOG_DEBUG(log_file, "top right");
                    v_x = GET(vx, vx->nx - 1, vx->ny - 1);
                }
            }

            // vy
            float v_y = 0;
            if (x_int > 0) {
                x_1 = (i);
                x_2 = (i + 1);
                y_1 = (j - 1);
                y_2 = (j);
            } else {
                x_1 = (i - 1);
                x_2 = (i);
                y_1 = (j - 1);
                y_2 = (j);
            }

            if (!(x_1 < 0 || y_1 < 0 || x_2 >= (int)vy->nx ||
                  y_2 >= (int)vy->ny))
                v_y = interpolate_bilinear(x, y, (x_1 + vy->x_internal) * dx,
                                           (y_1 + vy->y_internal) * dx,
                                           GET(vy, x_1, y_1), GET(vy, x_2, y_1),
                                           GET(vy, x_1, y_2), GET(vy, x_2, y_2),
                                           dx, dx);
            else {
                LOG_DEBUG(log_file, "edge case")
                // 8 possibilities

                // left side
                if (x_1 < 0 && y_1 >= 0 && y_2 < (int)vy->ny) {
                    LOG_DEBUG(log_file, "left side")
                    v_y = GET(vy, 0, y_1) * (1 - (y - y_1 * dx) / (dx)) +
                          GET(vy, 0, y_2) * (y - y_1 * dx) / dx;
                }

                // right side
                if (x_2 >= (int)vy->nx && y_1 >= 0 && y_2 < (int)vy->ny) {
                    LOG_DEBUG(log_file, "right side")
                    v_y =
                        GET(vy, vy->nx - 1, y_1) * (1 - (y - y_1 * dx) / (dx)) +
                        GET(vy, vy->nx - 1, y_2) * (y - y_1 * dx) / dx;
                }

                // bottom side
                if (y_1 < 0 && x_1 >= 0 && x_2 < (int)vy->nx) {
                    LOG_DEBUG(log_file, "bottom side");
                    v_y = GET(vy, x_1, 0) * (1 - (x - x_1 * dx) / (dx)) +
                          GET(vy, x_2, 0) * (x - x_1 * dx) / dx;
                }

                // top side
                if (y_2 >= (int)vy->ny && x_1 >= 0 && x_2 < (int)vy->nx) {
                    LOG_DEBUG(log_file, "top side");
                    v_y =
                        GET(vy, x_1, vy->ny - 1) * (1 - (x - x_1 * dx) / (dx)) +
                        GET(vy, x_2, vy->ny - 1) * (x - x_1 * dx) / dx;
                }

                // bottom left
                if (x_1 < 0 && y_1 < 0) {
                    LOG_DEBUG(log_file, "botttom left");
                    v_y = GET(vy, 0, 0);
                }

                // top left
                if (x_1 < 0 && y_2 >= (int)vy->ny) {
                    LOG_DEBUG(log_file, "top left");
                    v_y = GET(vy, 0, vy->ny - 1);
                }

                // bottom right
                if (x_2 >= (int)vy->nx && y_1 < 0) {
                    LOG_DEBUG(log_file, "bottom right");
                    v_y = GET(vy, vy->nx - 1, 0);
                }

                // top right
                if (x_2 >= (int)vy->nx && y_2 >= (int)vy->ny) {
                    LOG_DEBUG(log_file, "top right");
                    v_y = GET(vy, vy->nx - 1, vy->ny - 1);
                }
            }
            // xp
            float xp_x = x - dt * v_x;
            float xp_y = y - dt * v_y;

            int xp = (int)(((xp_x) / dx - 0.5) + 1);
            int yp = (int)(((xp_y) / dx - 0.5) + 1);
            float x_int_interp = (xp_x / dx) - xp;
            float y_int_interp = (xp_y / dx) - yp;

            // Get q at xp
            // vx
            float q_interp = 0;
            if (x_int_interp > x_int) {
                x_1 = xp;
                x_2 = xp + 1;
            } else {
                x_1 = xp - 1;
                x_2 = xp;
            }

            if (y_int_interp > y_int) {
                y_1 = yp;
                y_2 = yp + 1;
            } else {
                y_1 = yp - 1;
                y_2 = yp;
            }

            if (!(x_1 < 0 || y_1 < 0 || x_2 >= (int)q_n->nx ||
                  y_2 >= (int)q_n->ny))
                q_interp = interpolate_bilinear(
                    xp_x, xp_y, (x_1 + x_int) * dx, (y_1 + y_int) * dx,
                    GET(q_n, x_1, y_1), GET(q_n, x_2, y_1), GET(q_n, x_1, y_2),
                    GET(q_n, x_2, y_2), dx, dx);
            else {
                LOG_DEBUG(log_file, "edge case")
                // 8 possibilities

                // left side
                if (x_1 < 0 && y_1 >= 0 && y_2 < (int)q_n->ny) {
                    LOG_DEBUG(log_file, "left side")
                    q_interp =
                        GET(q_n, 0, y_1) * (1 - (xp_y - y_1 * dx) / (dx)) +
                        GET(q_n, 0, y_2) * (xp_y - y_1 * dx) / dx;
                }

                // right side
                if (x_2 >= (int)q_n->nx && y_1 >= 0 && y_2 < (int)q_n->ny) {
                    LOG_DEBUG(log_file, "right side")
                    q_interp =
                        GET(q_n, q_n->nx - 1, y_1) *
                            (1 - (xp_y - y_1 * dx) / (dx)) +
                        GET(q_n, q_n->nx - 1, y_2) * (xp_y - y_1 * dx) / dx;
                }

                // bottom side
                if (y_1 < 0 && x_1 >= 0 && x_2 < (int)q_n->nx) {
                    LOG_DEBUG(log_file, "bottom side");
                    q_interp =
                        GET(q_n, x_1, 0) * (1 - (xp_x - x_1 * dx) / (dx)) +
                        GET(q_n, x_2, 0) * (xp_x - x_1 * dx) / dx;
                }

                // top side
                if (y_2 >= (int)q_n->ny && x_1 >= 0 && x_2 < (int)q_n->nx) {
                    LOG_DEBUG(log_file, "top side");
                    q_interp =
                        GET(q_n, x_1, q_n->ny - 1) *
                            (1 - (xp_x - x_1 * dx) / (dx)) +
                        GET(q_n, x_2, q_n->ny - 1) * (xp_x - x_1 * dx) / dx;
                }

                // bottom left
                if (x_1 < 0 && y_1 < 0) {
                    LOG_DEBUG(log_file, "botttom left");
                    q_interp = GET(q_n, 0, 0);
                }

                // top left
                if (x_1 < 0 && y_2 >= (int)q_n->ny) {
                    LOG_DEBUG(log_file, "top left");
                    q_interp = GET(q_n, 0, q_n->ny - 1);
                }

                // bottom right
                if (x_2 >= (int)q_n->nx && y_1 < 0) {
                    LOG_DEBUG(log_file, "bottom right");
                    q_interp = GET(q_n, q_n->nx - 1, 0);
                }

                // top right
                if (x_2 >= (int)q_n->nx && y_2 >= (int)q_n->ny) {
                    LOG_DEBUG(log_file, "top right");
                    q_interp = GET(q_n, q_n->nx - 1, q_n->ny - 1);
                }
            }

            SET(q_n1, i, j, q_interp);
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

    if (check_params(data, log_file)) {
        LOG_ERR(log_file, "Problem checking the input parameters")
        return EXIT_FAILURE;
    }

    // Getting base params
    const unsigned int nx = data["grid"][0], ny = data["grid"][1];
    const float dx = data["space_steps"];
    const int sampling_rate = data["sampling_rate"];

    // Initialising the fields
    scalar_field *vx =
        scalar_field_init("vx", nx + 1, ny, 0.5, 0, dx, log_file);
    scalar_field *vy =
        scalar_field_init("vy", nx, ny + 1, 0, 0.5, dx, log_file);
    scalar_field *p = scalar_field_init("p", nx, ny, 0, 0, dx, log_file);
    if (!vx || !vy || !p) {
        LOG_ERR(log_file, "An error occured initializing fields.");
        return EXIT_FAILURE;
    }

    // Applying the initial conditions
    apply_initial_condition(vx, data, "ic_vx", log_file);
    apply_initial_condition(vy, data, "ic_vy", log_file);
    apply_initial_condition(p, data, "ic_p", log_file);

    float dt = 0.1;
    if (data.contains("delta_t"))
        dt = data["delta_t"];

    unsigned int nt = 10;
    if (data.contains("nt"))
        nt = data["nt"];

    scalar_field *temp_vx = scalar_field_copy(vx, log_file);
    scalar_field *temp_vy = scalar_field_copy(vy, log_file);
    scalar_field *temp_p = scalar_field_copy(p, log_file);

    // size_t size_vx = vx->nx * vx->ny * sizeof(float);
    // size_t size_vy = vy->nx * vy->ny * sizeof(float);
    // size_t size_p = p->nx * p->ny * sizeof(float);

    // Main time loop
    bool inverted = false;
    for (unsigned int i = 0; i < nt; i++) {
        LOG_INFO(log_file, "Starting time loop " << i << " out of " << nt);
        // project
        // save files
        if (sampling_rate && !(i % sampling_rate)) {
            write_scalar_vtk(vx, i, 0, log_file);
            write_scalar_vtk(vy, i, 0, log_file);
            write_scalar_vtk(p, i, 0, log_file);
        }

        // advect
        advect(vx, vy, dt, vx, temp_vx, log_file);
        advect(vx, vy, dt, vy, temp_vy, log_file);
        advect(vx, vy, dt, p, temp_p, log_file);

        inverted = !inverted;
        scalar_field *invert_vx = vx;
        scalar_field *invert_vy = vy;
        scalar_field *invert_p = p;

        vx = temp_vx;
        vy = temp_vy;
        p = temp_p;

        temp_vx = invert_vx;
        temp_vy = invert_vy;
        temp_p = invert_p;

        // memcpy(vx->values, temp_vx->values, size_vx);
        // memcpy(vy->values, temp_vy->values, size_vy);
        // memcpy(p->values, temp_p->values, size_p);
    }

    write_manifest_vtk(vx->name, dt, nt, 1, 1, 0, log_file);
    write_manifest_vtk(vy->name, dt, nt, 1, 1, 0, log_file);
    write_manifest_vtk(p->name, dt, nt, sampling_rate, 1, 0, log_file);

    // As we're not using objects, we need this
    scalar_field_free(vx, log_file);
    scalar_field_free(vy, log_file);
    scalar_field_free(p, log_file);

    scalar_field_free(temp_vx, log_file);
    scalar_field_free(temp_vy, log_file);
    scalar_field_free(temp_p, log_file);

    return EXIT_SUCCESS;
}