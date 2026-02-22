
#ifndef __SOLVER_CONDITIONS__
#define __SOLVER_CONDITIONS__
#include <fstream>
#include "data.hpp"
#include "nlohmann/json.hpp"
using json = nlohmann::json;
enum CELL_TYPE { LIQUID, SOLID };
/*
 @brief Applies some initial conditions to the scalar field
 @param scalar_field: the field to which apply the conditions (the domain)
 @param data: the full input json
 @param condition_name: the name of the initial condition in the json
 @param log_file: the log file
*/
int initialize_domain(scalar_field *field, json &data,
                      std::string condition_name, std::ofstream &log_file);
<<<<<<< HEAD
// Initializes the speed fields
int initialize_speed(scalar_field *field, scalar_field *dom, json &data,
                     std::string condition_name, std::ofstream &log_file);
=======

// Initializes the speed fields
int initialize_speed(scalar_field *field, scalar_field *dom, json &data,
                     std::string condition_name, std::ofstream &log_file);

>>>>>>> origin/quentin
#endif