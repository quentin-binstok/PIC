#ifndef __SOLVER_DATA__
#define __SOLVER_DATA__

#include <string>

typedef struct _scalar_field {
    std::string name;
    int nx, ny;
    float dx, dy;
    float *values;
} scalar_field;

typedef struct _particle_field {
    std::string name;
    int N;
    float *xyz; // soit 2D soit 3D, each particle's position aligned
    float *velocity; // idem
    int *id;
} particle_field;

scalar_field *scalar_field_init(const std::string name, const unsigned int nx, const unsigned int ny, const float dx, const float dy, std::ofstream& log_file);
void scalar_field_free(scalar_field *field, std::ofstream& log_file);

int write_manifest_vtk(std::string name, double dt, int nt, int sampling_rate,
                       int numranks, bool vtp, std::ofstream& log_file);

int write_data_vtk(scalar_field *data, int step, int rank, bool vtp, std::ofstream& log_file);

int write_particles_vtp(const particle_field* field,
                        const int step,
                        const int rank,
                      const int ndim, std::ofstream& log_file);
#endif