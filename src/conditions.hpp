
#ifndef __SOLVER_CONDITIONS__
#define __SOLVER_CONDITIONS__
#include <fstream>
#include "data.hpp"
#include "nlohmann/json.hpp"
using json = nlohmann::json;
enum CELL_TYPE {LIQUID, SOLID, AIR, DIRICHLET};
/*
 @brief Applies some initial conditions to the scalar field
 @param scalar_field: the field to which apply the conditions (the domain)
 @param data: the full input json
 @param condition_name: the name of the initial condition in the json
 @param log_file: the log file
*/
int initialize_domain(scalar_field *field, json &data,
                      std::string condition_name, std::ofstream &log_file);

// Initializes the speed fields
int initialize_speed(scalar_field *field, scalar_field *dom, json &data,
                     std::string condition_name, std::ofstream &log_file);

int create_circle(scalar_field *dom, std::string condition_name, json &data,
                  std::ofstream &log_file);

// Sets boundary conditions
int boundary_condition(scalar_field *vx, scalar_field *vy, scalar_field *dom, json &data,
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

#endif