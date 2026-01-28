#ifndef __SOLVER_DATA__
#define __SOLVER_DATA__

#include <string>

typedef struct _scalar_field {
    std::string name;
    int nx, ny;
    float dx, dy;
    float *values;
} scalar_field;

int write_manifest_vtk(std::string name, double dt, int nt, int sampling_rate,
                       int numranks, std::iostream log_file);

int write_data_vtk(scalar_field *data, int step, int rank, bool vtp, std::ostream log_file);

int write_particles_vtp(const char *name,
                        int step,
                        int rank,
                        int N,
                        const float *xyz,
                        const float *velocity,
                        const int   *id, 
                      const int ndim, std::iostream log_file);

#endif