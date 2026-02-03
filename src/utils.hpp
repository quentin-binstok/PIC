#ifndef __SOLVER_UTILS__
#define __SOLVER_UTILS__

#define LOG_INFO(log_file, msg) log_file << "[INFO] " << msg << std::endl;
#define LOG_WARN(log_file, msg) log_file << "[WARN] " << msg << std::endl;
#define LOG_ERR(log_file, msg) log_file << "[ERROR] " << msg << std::endl;

float interpolate_bilinear(int x, int y, int x1, int y1, float q11, 
                           float q21, float q12, float q22,
                           float Dx, float Dy);

#endif