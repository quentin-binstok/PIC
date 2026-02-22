
#include "utils.hpp"
float interpolate_bilinear(float x, float y, float x1, float y1, float q11,
                           float q21, float q12, float q22, float Dx,
                           float Dy) {
    // x1,y1 is the bottom-left corner of the cell, Dx and Dy are the cell sizes
    // q11, q21, q12, q22 are the values at the corners (11 is bottom-left)
    float dx = x - x1;
    float dy = y - y1;
    float value = (1 - (dx / Dx) - (dy / Dy) + ((dx * dy) / (Dx * Dy))) * q11 +
                  (dx / Dx - ((dx * dy) / (Dx * Dy))) * q21 +
                  (dy / Dy - ((dx * dy) / (Dx * Dy))) * q12 +
                  ((dx * dy) / (Dx * Dy)) * q22;
    return value;
}