#ifndef __SOLVER_DATA__
#define __SOLVER_DATA__

#include <string>

#define GET(data, i, j) ((data)->values[(data)->nx * (j) + (i)])
#define SET(data, i, j, val) ((data)->values[(data)->nx * (j) + (i)] = (val))

/*
A structure to store scalar fields
name: the name of the field
nx, ny: the size of the grid
dx: the step
values: the array of values
*/
typedef struct _scalar_field {
    std::string name;
    int nx, ny;
    float dx;
    float *values;
} scalar_field;

/*
A structure to store particles
name: the name of the particle field
N: the number of particles
xyz: the array of particle positions
velocity: the array of particle velocities
id: the ids of the particles, to identify them in paraview
*/
typedef struct _particle_field {
    std::string name;
    int N;
    float *xyz;      // soit 2D soit 3D, each particle's position aligned
    float *velocity; // idem
    int *id;
} particle_field;

/*
 @brief Initialises a scalar field
 @param name: the name, useful when writing files
 @param nx, ny: the extend of the grid
 @param dx: the step
 @param log_file: the log file
*/
scalar_field *scalar_field_init(const std::string name, const unsigned int nx,
                                const unsigned int ny, const float dx,
                                std::ofstream &log_file);

/*
 @brief Frees a scalar field
 @param field: the field
 @param log_file: the log file
*/
void scalar_field_free(scalar_field *field, std::ofstream &log_file);

int write_manifest_vtk(std::string name, double dt, int nt, int sampling_rate,
                       int numranks, bool vtp, std::ofstream &log_file);

int write_scalar_vtk(scalar_field *data, int step, int rank,
                     std::ofstream &log_file);

int write_particles_vtp(const particle_field *field, const int step,
                        const int rank, const int ndim,
                        std::ofstream &log_file);
#endif