
#ifndef __SOLVER_CONDITIONS__
#define __SOLVER_CONDITIONS__
#include "data.hpp"
#include "nlohmann/json.hpp"
#include <fstream>

using json = nlohmann::json;
enum CELL_TYPE { LIQUID, SOLID, AIR, DIRICHLET };

/*
 @brief Applies some initial condition "condition_name" to the domain
 @param scalar_field: the field to which apply the conditions (the domain)
 @param data: the full input json
 @param condition_name: the name of the initial condition in the json
 @param log_file: the log file
*/
int initialize_domain(scalar_field *field, json &data,
                      std::string condition_name, std::ofstream &log_file);

/*
 @brief Specialized function to apply initial conditions to the velocity fields
*/
int initialize_speed(scalar_field *field, scalar_field *dom, json &data,
                     std::string condition_name, std::ofstream &log_file);

/*
 @brief Applies speed & dom BCs
*/
int boundary_condition(scalar_field *vx, scalar_field *vy, scalar_field *dom,
                       std::vector<float> &speed_condition, json &data,
                       std::string condition_name, std::ofstream &log_file);

// Initializes an arbitrary field
int initialize_field(scalar_field *field, std::string condition_name,
                     json &data, std::ofstream &log);

// Function to handle the user defined fields
int get_fields(user_fields *fields, json &data, std::ofstream &log);

/*
 @brief puts the circles in the domain
 @param: dom, the domain
 @param condition_name: the name of the condition handling that
 @param data: the whole input json
 @param: log_file
 @returns: success or failure
*/
int create_circle(scalar_field *dom, std::string condition_name, json &data,
                  std::ofstream &log_file);

int initialize_taylor_green_vortex( scalar_field *vx, scalar_field *vy, scalar_field *dom, json &data,
                       std::string condition_name, std::ofstream &log_file);

#endif