
#ifndef __SOLVER_UTILS__
#define __SOLVER_UTILS__
#include <fstream>
#define LOG_INFO(log_file, msg) log_file << "[INFO] " << msg << std::endl;
#define LOG_WARN(log_file, msg) log_file << "[WARN] " << msg << std::endl;
#define LOG_ERR(log_file, msg) log_file << "[ERROR] " << msg << std::endl;
/*
 @brief Bilinear intepolation of q
 @param x, y: the absolute coordinates at which we want q
 @param x1, y1: the absolute coordinates of the inferior left corner of domain
    of interpolation
 @param q11, q21, q12, q22: the values of q at the corners
 @param Dx, Dy: the space steps
*/
float interpolate_bilinear(float x, float y, float x1, float y1, float q11,
                           float q21, float q12, float q22, float Dx, float Dy,
                           std::ofstream &log_file);
#endif