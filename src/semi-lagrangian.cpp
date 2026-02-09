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

    for (unsigned int j = 1; j < ny - 1; j++) {
        for (unsigned int i = 1; i < nx - 1; i++) {
            LOG_INFO(log_file,
                     "Starting advection at (" << i << "," << j << ")");
            /*
            1. Interpolate speed field at the point
            2. Get the xp
            3. Replace the value (i need an intermediate field)
            */

            // Interpolation of the speed field

            // coords where we need the speed field
            float x = (i + x_int) * dx, y = (j + y_int) * dx;

            // vx
            float v_x;
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

            LOG_INFO(log_file, "Interpolating vx");
            v_x = interpolate_bilinear(
                x, y, (x_1 + vx->x_internal) * dx, (y_1 + vx->y_internal) * dx,
                GET(vx, x_1, y_1), GET(vx, x_2, y_1), GET(vx, x_1, y_2),
                GET(vx, x_2, y_2), dx, dx, log_file);
            LOG_INFO(log_file, "vx = " << v_x);

            // vy
            float v_y;
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

            LOG_INFO(log_file, "Interpolating vy")
            v_y = interpolate_bilinear(
                x, y, (x_1 + vy->x_internal) * dx, (y_1 + vy->y_internal) * dx,
                GET(vy, x_1, y_1), GET(vy, x_2, y_1), GET(vy, x_1, y_2),
                GET(vy, x_2, y_2), dx, dx, log_file);
            LOG_INFO(log_file, "vy = " << v_y);

            // xp
            float xp_x = x - dt * v_x;
            float xp_y = y - dt * v_y;

            LOG_INFO(log_file, "(xp_x, xp_y) " << xp_x << " " << xp_y);

            int xp = (int)(((xp_x) / dx - 0.5) + 1);
            int yp = (int)(((xp_y) / dx - 0.5) + 1);
            LOG_INFO(log_file, "(xp, yp) " << xp << " " << yp);
            float x_int_interp = (xp_x / dx) - xp;
            float y_int_interp = (xp_y / dx) - yp;

            // Get q at xp
            // vx
            float q_interp;
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

            if (x_1 < 0 || y_1 < 0 || x_2 >= (int)nx || y_2 >= (int)nx)
                continue;

            LOG_INFO(log_file, "Interpolating q " << x_1 << x_2 << y_1 << y_2);
            q_interp = interpolate_bilinear(
                xp_x, xp_y, (x_1 + x_int) * dx, (y_1 + y_int) * dx,
                GET(q_n, x_1, y_1), GET(q_n, x_2, y_1), GET(q_n, x_1, y_2),
                GET(q_n, x_2, y_2), dx, dx, log_file);
            LOG_INFO(log_file, "q_interp " << q_interp);

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

    // This is just to have a whole pipeline
    // write_scalar_vtk(vx, 0, 0, log_file);
    // write_scalar_vtk(vy, 0, 0, log_file);
    // write_scalar_vtk(p, 0, 0, log_file);

    float dt = 0.1;
    if (data.contains("delta_t"))
        dt = data["delta_t"];

    unsigned int nt = 10;
    if (data.contains("nt"))
        nt = data["nt"];

    scalar_field *temp_vx = scalar_field_copy(vx, log_file);
    scalar_field *temp_vy = scalar_field_copy(vy, log_file);
    scalar_field *temp_p = scalar_field_copy(p, log_file);

    size_t size_vx = vx->nx * vx->ny * sizeof(float);
    size_t size_vy = vy->nx * vy->ny * sizeof(float);
    size_t size_p = p->nx * p->ny * sizeof(float);

    // Main time loop
    for (unsigned int i = 0; i < nt; i++) {
        LOG_INFO(log_file, "Starting time loop " << i << " out of " << nt);
        // project
        // save files
        write_scalar_vtk(vx, i, 0, log_file);
        write_scalar_vtk(vy, i, 0, log_file);
        write_scalar_vtk(p, i, 0, log_file);

        // advect
        advect(vx, vy, dt, vx, temp_vx, log_file);
        advect(vx, vy, dt, vy, temp_vy, log_file);
        advect(vx, vy, dt, p, temp_p, log_file);

        memcpy(vx->values, temp_vx->values, size_vx);
        memcpy(vy->values, temp_vy->values, size_vy);
        memcpy(p->values, temp_p->values, size_p);
    }

    write_manifest_vtk(vx->name, dt, nt, 1, 1, 0, log_file);
    write_manifest_vtk(vy->name, dt, nt, 1, 1, 0, log_file);
    write_manifest_vtk(p->name, dt, nt, 1, 1, 0, log_file);

    // As we're not using objects, we need this
    scalar_field_free(vx, log_file);
    scalar_field_free(vy, log_file);
    scalar_field_free(p, log_file);

    return EXIT_SUCCESS;
}