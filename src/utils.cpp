#include <cstdlib>
#include <istream>
#include <fstream>
#include <string>
#include <cinttypes>
#include "data.hpp"
#include "utils.hpp"

float bilinear_interpolate(float x, float y, int x1, int y1,
                           float Q11, float Q12, float Q21, float Q22, 
                           float Dx, float Dy) {
    // x1, y1 is the bottom-left corner, Q are the values at the corners, Dx and Dy are the cell sizes
    float x2 = x1 + Dx;
    float y2 = y1 + Dy;
    float fxy1 = ((x2 - x) / (Dx)) * Q11 + ((x - x1) / (Dx)) * Q21;
    float fxy2 = ((x2 - x) / (Dx)) * Q12 + ((x - x1) / (Dx)) * Q22;
    return ((y2 - y) / (Dy)) * fxy1 + ((y - y1) / (Dy)) * fxy2;
}

int init_scalar_field(scalar_field* field, const std::string& name,
                      const int nx, const int ny,
                      const double dx, const double dy,
                      std::ofstream& log_file) {
    field->name = name;
    field->nx = nx;
    field->ny = ny;
    field->dx = dx;
    field->dy = dy;

    field->values = (float*)calloc(field->nx * field->ny, sizeof(float));
    if (!field->values) {
        LOG_ERR(log_file, "Could not allocate scalar field " << name);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
void free_data(scalar_field* field) { free(field->values);}