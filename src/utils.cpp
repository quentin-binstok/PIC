#include "utils.hpp"
#include "conditions.hpp"
#include "data.hpp"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

float interpolate_bilinear(float x, float y, float x1, float y1, float q11,
                           float q21, float q12, float q22, float Dx,
                           float Dy) {
    // x1,y1 is the bottom-left corner of the cell, Dx and Dy are the cell sizes
    // q11, q21, q12, q22 are the values at the corners (11 is bottom-left)
    float dx = x - x1;
    float dy = y - y1;
    float value = (1 - (dx / Dx) - (dy / Dy) + ((dx * dy) / (Dx * Dy))) * q11 +
                  (dx / Dx - ((dx * dy) / (Dx * Dy))) * q21 +
                  (dy / Dy - ((dx * dy) / (Dx * Dy))) * q12 +
                  ((dx * dy) / (Dx * Dy)) * q22;
    return value;
}

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
 @brief interpolates the speed at x, y
 @params: v_y, v_y are the values to which the speed will be written
*/
int get_speed(float *v_x, float *v_y, float x, float y, scalar_field *vx,
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

/*
 @brief computes the divergence of the velocity field
 @param vx, vy: the velocity field
 @param div: the divergence field
 @param dx: the grid spacing
 @return the maximum divergence in the field, for logging purposes
*/
int divergence(scalar_field *vx, scalar_field *vy, scalar_field *div,
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
 @brief projects the velocity field to make it divergence free
 @param p: the pressure field
 @param vx, vy: the velocity field
 @param dt: the time step
 @param rho: the density
*/
int project_velocity(scalar_field *p, scalar_field *vx, scalar_field *vy,
                     scalar_field *dom, float dx, float dt, float rho,
                     std::ofstream &log_file,
                     std::vector<float> &speed_condition) {
    LOG_INFO(log_file, "Projecting the velocity field")
    int vx_nx = vx->nx;
    int vx_ny = vx->ny;
    int vy_nx = vy->nx;
    int vy_ny = vy->ny;
#pragma omp parallel for collapse(2)
    for (int j = 0; j < vx_ny; j++) {
        for (int i = 0; i < vx_nx; i++) {
            if (GET(dom, i, j) == SOLID ||
                (i != vx_nx - 1 && GET(dom, i + 1, j) == SOLID)) {
                SET(vx, i, j, 0.0);
                continue;
            }
            if (GET(dom, i, j) == DIRICHLET) {
                SET(vx, i, j, 0);
                if (i == 0) {
                    float v = speed_condition[0];
                    SET(vx, i, j, v);
                }

                if (i == vx_nx - 1) {
                    float v = speed_condition[1];
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
            if (GET(dom, i, j) == SOLID ||
                (j != vy_ny - 1 && GET(dom, i, j + 1) == SOLID)) {
                SET(vy, i, j, 0.0);
                continue;
            }
            if (GET(dom, i, j) == DIRICHLET) {
                SET(vy, i, j, 0);
                if (j == 0) {
                    float v = speed_condition[3];
                    SET(vy, i, j, v);
                }

                if (j == vy_ny - 1) {
                    float v = speed_condition[2];
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

    // Free-slip on left/right walls
    for (int j = 0; j < vy_ny; j++) {
        if (GET(dom, 0, vy_ny / 2) == SOLID)
            SET(vy, 0, j, GET(vy, 1, j));
        if (GET(dom, vy_nx - 1, vy_ny / 2) == SOLID)
            SET(vy, vy_nx - 1, j, GET(vy, vy_nx - 2, j));
    }

    // for vx on horizontal walls
    for (int i = 0; i < vx_nx; i++) {
        if (GET(dom, vx_nx / 2, 0) == SOLID)
            SET(vx, i, 0, GET(vx, i, 1));
        if (GET(dom, vx_nx / 2, vx_ny - 1) == SOLID)
            SET(vx, i, vx_ny - 1, GET(vx, i, vx_ny - 2));
    }

    return EXIT_SUCCESS;
}


float volume(scalar_field *dom, float dx) {
    int nx = dom->nx;
    int ny = dom->ny;
    float volume = 0;
#pragma omp parallel for collapse(2) reduction(+ : volume)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if (GET(dom, i, j) == LIQUID) {
                volume += dx * dx;
            }
        }
    }
    return volume;
}

float free_surface_area(scalar_field *dom, float dx) {
    int nx = dom->nx;
    int ny = dom->ny;
    float area = 0;
#pragma omp parallel for collapse(2) reduction(+ : area)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if (GET(dom, i, j) == LIQUID) {
                if (i > 0 && GET(dom, i - 1, j) == AIR)
                    area += dx;
                if (i < nx - 1 && GET(dom, i + 1, j) == AIR)
                    area += dx;
                if (j > 0 && GET(dom, i, j - 1) == AIR)
                    area += dx;
                if (j < ny - 1 && GET(dom, i, j + 1) == AIR)
                    area += dx;
            }
        }
    }
    return area;
}

float depth(scalar_field *dom, int idx, float dx) {
    int ny = dom->ny;
    int depth = 0;
    for (int j = 1; j < ny; j++) {
        if (GET(dom, idx, j) == LIQUID) {
            depth++;
        } else {
            break;
        }
    }
    return depth * dx;
}
