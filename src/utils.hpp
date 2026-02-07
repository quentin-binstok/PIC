#ifndef __SOLVER_UTILS__
#define __SOLVER_UTILS__

#define LOG_INFO(log_file, msg) log_file << "[INFO] " << msg << std::endl;
#define LOG_WARN(log_file, msg) log_file << "[WARN] " << msg << std::endl;
#define LOG_ERR(log_file, msg) log_file << "[ERROR] " << msg << std::endl;

float bilinear_interpolate(float x, float y, int x1, int y1,
                           float Q11, float Q12, float Q21, float Q22, 
                           float Dx, float Dy);

int init_scalar_field(scalar_field* field, const std::string& name,
                      const int nx, const int ny,
                      const double dx, const double dy,
                      std::ofstream& log_file);

void free_data(scalar_field* field);

#endif