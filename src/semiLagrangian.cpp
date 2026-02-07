#include <array>
#include <cstdlib>
#include <fstream>
#include "data.hpp"
#include "utils.hpp"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

int check_params(json& data, std::ofstream& log_file) {
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
            LOG_ERR(log_file, "Elements from \"grid\" cannot be zero or negative");
            return EXIT_FAILURE;
        }
    }

    // space_steps
    if (data.contains("space_steps")) {
        if (data["space_steps"].type() != json::value_t::array) {
            LOG_ERR(log_file, "\"space_steps\" not provided as array");
            return EXIT_FAILURE;
        }

        if (data["space_steps"].size() != 2) {
            LOG_ERR(log_file, "\"space_steps\" not 2 elements long");
            return EXIT_FAILURE;
        }

        if (data["space_steps"][0] <= 0 || data["space_steps"][1] <= 0) {
            LOG_ERR(log_file, "Elements from \"space_steps\" cannot be zero or negative");
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}

int solver_semi_lagrangian(json& data, std::ofstream& log_file) {
    LOG_INFO(log_file, "Starting the semi-lagrangian solver");

    if (check_params(data, log_file)) {
        LOG_ERR(log_file, "Problem checking the input parameters");
        return EXIT_FAILURE;
    }

    // Getting base params
    const unsigned int nx = data["grid"][0], ny = data["grid"][1];
    const float dx = data["space_steps"][0], dy = data["space_steps"][1];
    const float dt = data["time_step"];

    scalar_field *vx = scalar_field_init("vx", nx + 1, ny, dx, dy,log_file);
    scalar_field *vy = scalar_field_init("vy", nx, ny + 1, dx, dy,log_file);
    scalar_field *p = scalar_field_init("p", nx, ny, dx, dy,log_file);

    if (!vx || !vy || !p) {
        LOG_ERR(log_file, "An error occured initializing fields.");
        return EXIT_FAILURE;
    }
    float rho = 1.0;
    float alpha = dt/(rho * dx);

    LOG_INFO(log_file, "Setting 1 values");

    for (int j = 0; j < p->ny; j++) {
        p->values[p->nx * j + 1] = 1.0;
    }

    LOG_INFO(log_file, "Writing first vtk");
    write_data_vtk(p, 0, 0, log_file);
    write_data_vtk(vx, 0, 0, log_file);
    write_data_vtk(vy, 0, 0, log_file);

    LOG_INFO(log_file, "starting iterations");

    int steps = 50;
    for (int n = 1; n < steps; n++) {
        for (int j = 1; j < p->ny - 1; j++) {
            for (int i = 1; i < p->nx - 1; i++) {
                p->values[p->nx * j + i] = (p->values[p->nx * j + (i-1)] + p->values[p->nx * j + (i+1)] 
                       + p->values[p->nx * (j-1) + i] + p->values[p->nx * (j+1) + i])/4
                       - (vx->values[vx->nx * j + (i+1)] - vx->values[vx->nx * j + i] 
                          + vy->values[vy->nx * (j+1) + i] - vy->values[vy->nx * j + i])/(4*alpha);
            }
        }
        for (int j = 1; j < vx->ny - 1; j++) {
            for (int i = 1; i < vx->nx - 1; i++) {
                vx->values[vx->nx * j + i] = vx->values[vx->nx * j + i] 
                    - alpha * (p->values[p->nx * j + (i+1)] - p->values[p->nx * j + i]);
                vy->values[vy->nx * j + i] = vy->values[vy->nx * j + i] 
                    - alpha * (p->values[p->nx * (j+1) + i] - p->values[p->nx * j + i]);
            }
        }

        for (int j = 1; j < vx->ny - 1; j++) {
            for (int i = 1; i < vx->nx - 1; i++) {
                vx->values[vx->nx * j + i] = vx->values[vx->nx * j + i] 
                    - alpha * (p->values[p->nx * j + (i+1)] - p->values[p->nx * j + i]);
                vy->values[vy->nx * j + i] = vy->values[vy->nx * j + i] 
                    - alpha * (p->values[p->nx * (j+1) + i] - p->values[p->nx * j + i]);
            }
        }

        for (int j = 1; j < vx->ny - 1; j++) {
            for (int i = 1; i < vx->nx - 1; i++) {
                float x = i - vx->values[vx->nx * j + i];
                float y = j - vy->values[vy->nx * j + i];

                float new_u = bilinear_interpolate(x, y, i, j,
                                               vx->values[vx->nx * (j-1) + (i-1)], vx->values[vx->nx * (j+1) + (i-1)],
                                               vx->values[vx->nx * (j-1) + (i+1)], vx->values[vx->nx * (j+1) + (i+1)],
                                               dx, dy);
                float new_v = bilinear_interpolate(x, y, i, j,
                                               vy->values[vy->nx * (j-1) + (i-1)], vy->values[vy->nx * (j+1) + (i-1)],
                                               vy->values[vy->nx * (j-1) + (i+1)], vy->values[vy->nx * (j+1) + (i+1)],
                                               dx, dy);

                vx->values[vx->nx * j + i] = new_u;
                vy->values[vy->nx * j + i] = new_v;
            }
        }
        write_data_vtk(p, n, 0, log_file);
        write_data_vtk(vx, n, 0, log_file);
        write_data_vtk(vy, n, 0, log_file);
    }
    write_manifest_vtk(p->name, (double)1, steps, 1, 1, false, log_file);
    write_manifest_vtk(vx->name, (double)1, steps, 1, 1, false, log_file);
    write_manifest_vtk(vy->name, (double)1, steps, 1, 1, false, log_file);
    
    scalar_field_free(vx, log_file);
    scalar_field_free(vy, log_file);
    scalar_field_free(p, log_file);

    return EXIT_SUCCESS;
}

/* void test_vtp(std::ofstream& log_file) {
    LOG_INFO(log_file, "Starting particle test")

    particle_field field;
    field.name = "test_field";
    field.N = 5;

    LOG_INFO(log_file, "Allocating field");

    field.xyz = (float*)calloc(field.N, 2 * sizeof(float));
    field.velocity = (float*)calloc(field.N, 2 * sizeof(float));
    field.id = (int*)calloc(field.N, sizeof(int));

    LOG_INFO(log_file, "Setting 0 values");
    for (int i = 0; i < field.N; i++) {
        field.xyz[2 * i] = i;
        field.xyz[2 * i + 1] = i;

        field.velocity[2 * i] = (float)i / 2;
        field.velocity[2 * i + 1] = -(float)i / 2 + 1;

        field.id[i] = i;
    }

    LOG_INFO(log_file, "Writing first vtp");
    write_particles_vtp(&field, 0, 0, 2, log_file);

    LOG_INFO(log_file, "starting iterations");
    int steps = 6;
    for (int i = 1; i < steps; i++) {
        for (int j = 0; j < field.N; j++){
            field.xyz[2 * j] += field.velocity[2 * j];
            field.xyz[2 * j + 1] += field.velocity[2 * j + 1];

            // field.velocity[2 * i] += -0.5;
            // field.velocity[2 * i + 1] += -0.5;

            write_particles_vtp(&field, i, 0, 2, log_file);
        }
    }

    write_manifest_vtk(field.name, (double)1, steps, 1, 1, true, log_file);

    LOG_INFO(log_file, "End of test");
    return;
} */