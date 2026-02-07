#include "conditions.hpp"
#include "data.hpp"
#include "nlohmann/json.hpp"
#include "utils.hpp"

#include <cstdlib>
#include <fstream>

using json = nlohmann::json;

/*
 @brief checks the validity of parameters (except boundary and initial
 conditions)
 @param data: the whole data json
 @param log_file: the log file
*/
int check_params(json &data, std::ofstream &log_file) {
    LOG_INFO(log_file, "Checking the input parameters");

    // grid
    if (data.contains("grid")) {
        if (data["grid"].type() != json::value_t::array) {
            LOG_ERR(log_file, "\"grid\" not provided as array");
            return EXIT_FAILURE;
        }

        if (data["grid"].size() != 2) {
            LOG_ERR(log_file, "\"grid\" not 2 elements long");
            return EXIT_FAILURE;
        }

        if (data["grid"][0] <= 0 || data["grid"][1] <= 0) {
            LOG_ERR(log_file,
                    "Elements from \"grid\" cannot be zero or negative");
            return EXIT_FAILURE;
        }
    }

    // space_steps
    if (data.contains("space_steps")) {
        if (data["space_steps"].type() != json::value_t::number_float) {
            LOG_ERR(log_file, "\"space_steps\" not provided as float");
            return EXIT_FAILURE;
        }

        if (data["space_steps"] <= 0) {
            LOG_ERR(log_file, "\"space_steps\" cannot be zero or negative");
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}

/*
 @brief the semi lagrangian solver
 @param data: the whole json
 @param log_file: the log file
*/
int solver_semi_lagrangian(json &data, std::ofstream &log_file) {
    LOG_INFO(log_file, "Starting the semi-lagrangian solver");

    if (check_params(data, log_file)) {
        LOG_ERR(log_file, "Problem checking the input parameters")
        return EXIT_FAILURE;
    }

    // Getting base params
    const unsigned int nx = data["grid"][0], ny = data["grid"][1];
    const float dx = data["space_steps"];

    // Initialising the fields
    scalar_field *vx = scalar_field_init("vx", nx + 1, ny, dx, log_file);
    scalar_field *vy = scalar_field_init("vy", nx, ny + 1, dx, log_file);
    scalar_field *p = scalar_field_init("p", nx, ny, dx, log_file);
    if (!vx || !vy || !p) {
        LOG_ERR(log_file, "An error occured initializing fields.");
        return EXIT_FAILURE;
    }

    // Applying the initial conditions
    apply_initial_condition(vx, data, "ic_vx", log_file);
    apply_initial_condition(vy, data, "ic_vy", log_file);
    apply_initial_condition(p, data, "ic_p", log_file);

    // This is just to have a whole pipeline
    write_scalar_vtk(vx, 0, 0, log_file);
    write_scalar_vtk(vy, 0, 0, log_file);
    write_scalar_vtk(p, 0, 0, log_file);

    // As we're not using objects, we need this
    scalar_field_free(vx, log_file);
    scalar_field_free(vy, log_file);
    scalar_field_free(p, log_file);

    return EXIT_SUCCESS;
}