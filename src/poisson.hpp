#ifndef __SOLVER_POISSON__
#define __SOLVER_POISSON__

#include "data.hpp"
#include "nlohmann/json.hpp"
#include "poisson.hpp"
#include "utils.hpp"
#include "conditions.hpp"

/*
 @brief computes the residual term of the pressure computation
 @params scarlar_field res: where the resulting Ax term will be stored
 @params same as everywhere
 @returns: the 2-norm of the residual
*/
float residual(scalar_field *p, scalar_field *vx, scalar_field *vy,
               scalar_field *div, scalar_field *dom, float rho, float dt,
               std::ofstream &log_file);

/*
 @brief solves the Poisson equation for the pressure using Jacobi
 iterations
 @param p: the pressure field
 @param temp_p: a temporary pressure field needed to work
 @param div: the divergence field
 @param vx, vy: the velocity field
 @param dom: the domain field
 @param tol: the tolerance at which to stop
 @param dt: the time step
 @param rho: the density
 @param max_iter: the max number of iterations
 @param first_looop: whether this is the first time loop or not
*/
int jacobi(scalar_field *p, scalar_field *temp_p, scalar_field *div,
           scalar_field *vx, scalar_field *vy, scalar_field *dom, float tol,
           float dt, float rho, int max_iter, bool first_loop,
           std::ofstream &log_file);

/*
 @brief solves the Poisson equation for the pressure using SOR
 iterations
 @param p: the pressure field
 @param div: the divergence field
 @param vx, vy: the velocity field
 @param dom: the domain field
 @param tol: the tolerance at which to stop
 @param dt: the time step
 @param rho: the density
 @param max_iter: the max number of iterations
*/
int sor(scalar_field *p, scalar_field *div, scalar_field *vx, scalar_field *vy,
        scalar_field *dom, float tol, float dt, float rho, int max_iter,
        std::ofstream &log_file);

#endif