#ifndef __SOLVER_DATA__
#define __SOLVER_DATA__
#include <filesystem>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

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
    float x_internal, y_internal;
    float dx;
    float *values;
} scalar_field;

/*
A structure to store the fields defined by the user
nb_fields: the number of created fields
names: their names
fields: the fields array
*/
typedef struct _user_fields {
    int nb_fields;
    std::vector<std::string> names;
    scalar_field **fields;
} user_fields;

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
    int next_id;
    std::vector<float> xyz; // soit 2D soit 3D, each particle's position aligned
    std::vector<float> velocity; // idem
    std::vector<float> T;
    std::vector<int> id;
    std::vector<float>
        ix; // for APIC, the interpolation of the velocity gradient
    std::vector<float> iy;
    std::vector<float> bx; // for APIC, the affine part of the velocity
    std::vector<float> by;
    /* std::vector<float> B; */
    /* std::vector<float> C; // for APIC affine matrix */
} particle_field;

struct RNG {
    std::mt19937 gen;
    std::uniform_real_distribution<float> jitter;
    std::uniform_real_distribution<float> prob;
    std::uniform_real_distribution<float> birth;

    RNG(float dx, float dt, std::optional<uint32_t> seed = std::nullopt)
        : gen(seed ? *seed : std::random_device{}()),
          jitter(-0.5f * dx, 0.5f * dx), prob(0.0f, 1.0f), birth(0.0f, dt) {}
};

/*
 @brief Initialises a scalar field
 @param name: the name, useful when writing files
 @param nx, ny: the extend of the grid
 @param x_internal, y_internal: the position of the field inside a single cell
 @param dx: the step
 @param log_file: the log file
*/
scalar_field *scalar_field_init(const std::string name, const unsigned int nx,
                                const unsigned int ny, const float x_internal,
                                const float y_internal, const float dx,
                                std::ofstream &log_file);

/*
 @brief copies the field, but not the values (not a deep copy!)
*/
scalar_field *scalar_field_copy(const scalar_field *field,
                                std::ofstream &log_file);

/*
 @brief Frees a scalar field
 @param field: the field
 @param log_file: the log file
*/
void scalar_field_free(scalar_field *field, std::ofstream &log_file);

/*
 @brief Initializing a particle field in 2D
 @param name: the name of the particle field
 @param N: the number of particles
 @param log_file: the log file
 @return: the particle field with resized std::vectors
*/
particle_field *particle_field_init_2D(const std::string name, const int N,
                                       std::ofstream &log_file);

/*
 @brief deep copy of the particle field
*/
particle_field *copy_particle_field(const particle_field *particles,
                                    std::ofstream &log_file);

// Frees the user fields
void user_field_free(user_fields *fields, std::ofstream &log);

int write_manifest_vtk(std::string name, double dt, int nt, int sampling_rate,
                       int numranks, bool vtp, std::ofstream &log_file,
                       fs::path work_dir);

int write_scalar_vtk(scalar_field *data, int step, int rank,
                     std::ofstream &log_file, fs::path work_dir);

int write_particles_vtp(const particle_field *field, const int step,
                        const int rank, const int ndim, std::ofstream &log_file,
                        fs::path work_dir);
#endif
