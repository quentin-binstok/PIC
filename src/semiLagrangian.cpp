#include "test.hpp"
#include "data.hpp"
#include "utils.hpp"
#include <fstream>
#include <cmath>

void semiLagrangian(std::ofstream& log_file){
    LOG_INFO(log_file, "Starting Lagrangian test")

    int nx = 10;
    int ny = 10;
    double dx = 1.0;
    double dy = 1.0;

    LOG_INFO(log_file, "Allocating field");

    scalar_field u, v, P;
    float alpha = 0.1;

    init_scalar_field(&u, "vel_x", nx, ny, dx, dy, log_file);
    init_scalar_field(&v, "vel_y", nx, ny, dx, dy, log_file);
    init_scalar_field(&P, "pressure", nx, ny, dx, dy, log_file);

    LOG_INFO(log_file, "Setting 0 values");

    for (int j = 0; j < P.ny; j++) {
        for (int i = 0; i < P.nx; i++) {
            P.values[P.nx * j + i] = 0.0;
            u.values[u.nx * j + i] = 1.0;
            v.values[v.nx * j + i] = 0.0;
        }
    }

    LOG_INFO(log_file, "Writing first vtk");
    write_data_vtk(&P, 0, 0, log_file);
    write_data_vtk(&u, 0, 0, log_file);
    write_data_vtk(&v, 0, 0, log_file);

    LOG_INFO(log_file, "starting iterations");

    int steps = 6;
    for (int n = 1; n < steps; n++) {
        for (int j = 1; j < P.ny - 1; j++) {
            for (int i = 1; i < P.nx - 1; i++) {
                P.values[P.nx * j + i] = (P.values[P.nx * j + (i-1)] + P.values[P.nx * j + (i+1)] 
                       + P.values[P.nx * (j-1) + i] + P.values[P.nx * (j+1) + i])/4
                       - (u.values[u.nx * j + (i+1)] - u.values[u.nx * j + i] 
                          + v.values[v.nx * (j+1) + i] - v.values[v.nx * j + i])/(4*alpha);
            }
        }
        for (int j = 1; j < u.ny - 1; j++) {
            for (int i = 1; i < u.nx - 1; i++) {
                u.values[u.nx * j + i] = u.values[u.nx * j + i] 
                    - alpha * (P.values[P.nx * j + (i+1)] - P.values[P.nx * j + i]);
                v.values[v.nx * j + i] = v.values[v.nx * j + i] 
                    - alpha * (P.values[P.nx * (j+1) + i] - P.values[P.nx * j + i]);
            }
        }

        for (int j = 1; j < u.ny - 1; j++) {
            for (int i = 1; i < u.nx - 1; i++) {
                u.values[u.nx * j + i] = u.values[u.nx * j + i] 
                    - alpha * (P.values[P.nx * j + (i+1)] - P.values[P.nx * j + i]);
                v.values[v.nx * j + i] = v.values[v.nx * j + i] 
                    - alpha * (P.values[P.nx * (j+1) + i] - P.values[P.nx * j + i]);
            }
        }

        for (int j = 1; j < u.ny - 1; j++) {
            for (int i = 1; i < u.nx - 1; i++) {
                float x = i - u.values[u.nx * j + i];
                float y = j - v.values[v.nx * j + i];

                float new_u = bilinear_interpolate(x, y, i, j,
                                               u.values[u.nx * (j-1) + (i-1)], u.values[u.nx * (j+1) + (i-1)],
                                               u.values[u.nx * (j-1) + (i+1)], u.values[u.nx * (j+1) + (i+1)],
                                               dx, dy);
                float new_v = bilinear_interpolate(x, y, i, j,
                                               v.values[v.nx * (j-1) + (i-1)], v.values[v.nx * (j+1) + (i-1)],
                                               v.values[v.nx * (j-1) + (i+1)], v.values[v.nx * (j+1) + (i+1)],
                                               dx, dy);

                u.values[u.nx * j + i] = new_u;
                v.values[v.nx * j + i] = new_v;
            }
        }
        write_data_vtk(&P, n, 0, log_file);
        write_data_vtk(&u, n, 0, log_file);
        write_data_vtk(&v, n, 0, log_file);
    }
    write_manifest_vtk(P.name, (double)1, steps, 1, 1, false, log_file);
    write_manifest_vtk(u.name, (double)1, steps, 1, 1, false, log_file);
    write_manifest_vtk(v.name, (double)1, steps, 1, 1, false, log_file);
    free_data(&u);
    free_data(&v);
    free_data(&P);
    LOG_INFO(log_file, "End of test");
    return;
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