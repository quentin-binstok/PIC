#include "test.hpp"
#include "data.hpp"
#include "utils.hpp"
#include <fstream>

void test_vtp(std::ofstream& log_file) {
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
}