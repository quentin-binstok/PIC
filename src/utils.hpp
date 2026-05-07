#ifndef __SOLVER_UTILS__
#define __SOLVER_UTILS__

#include <fstream>
#define LOG_INFO(log_file, msg) log_file << "[INFO] " << msg << std::endl;
#define LOG_WARN(log_file, msg) log_file << "[WARN] " << msg << std::endl;
#define LOG_ERR(log_file, msg) log_file << "[ERROR] " << msg << std::endl;
#ifndef NDEBUG
#define LOG_DEBUG(log_file, msg) log_file << "[DEBUG] " << msg << std::endl;
#else
#define LOG_DEBUG(log_file, msg) void();
#endif

#include "data.hpp"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

struct Metrics {
    int step;
    int particle_in_solid;
    int singularity_count;
    std::vector<std::string> headers;
    std::vector<float> values;
};

/*
 @brief Bilinear intepolation of q
 @param x, y: the absolute coordinates at which we want q
 @param x1, y1: the absolute coordinates of the inferior left corner of domain
    of interpolation
 @param q11, q21, q12, q22: the values of q at the corners
 @param Dx, Dy: the space steps
*/
float interpolate_bilinear(float x, float y, float x1, float y1, float q11,
                           float q21, float q12, float q22, float Dx, float Dy);
// overloading with int
float interpolate_bilinear(int x, int y, int x1, int y1, float q11, float q21,
                           float q12, float q22, float Dx, float Dy);

/*
 @brief checks the validity of parameters (except boundary and initial
 conditions)
 @param data: the whole data json
 @param log_file: the log file
 TODO: expand this with the new options
*/
int check_params(json &data, std::ofstream &log_file);

/*
 @brief interpolates the speed at x, y
 @params: v_y, v_y are the values to which the speed will be written
*/
int get_speed(float *v_x, float *v_y, float x, float y, scalar_field *vx,
              scalar_field *vy, std::ofstream &log_file);

/*
 @brief computes the divergence of the velocity field
 @param vx, vy: the velocity field
 @param div: the divergence field
 @param dx: the grid spacing
 @return the maximum divergence in the field, for logging purposes
*/
int divergence(scalar_field *vx, scalar_field *vy, scalar_field *div,
               scalar_field *dom, std::vector<float> speed_condition,
               std::ofstream &log_file);

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
                     std::vector<float> &speed_condition);

float volume(scalar_field *dom, float dx);
float free_surface_area(scalar_field *dom, float dx);
float depth(scalar_field *dom, int idx, float dx);

void write_metrics(std::ofstream &f, const Metrics &m);
void write_header(std::ofstream &f, const std::vector<std::string> &headers);
Metrics initialize_metrics(Metrics m, std::ofstream &log_file);
Metrics compute_metrics(scalar_field *dom, scalar_field *p, scalar_field *vx,
                        scalar_field *vy, scalar_field *div, float dx, int step,
                        int nt, Metrics m, const json &metric_data,
                        std::ofstream &log_file);
std::vector<std::string> build_headers(const json &metric_data);

void sine_surface(json &data, scalar_field *dom, std::ofstream &log_file);

#endif