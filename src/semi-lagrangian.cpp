#include <array>
#include <cstdlib>
#include <fstream>
#include "data.hpp"
#include "utils.hpp"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

int check_params(json& data, std::ofstream& log_file) {
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
            LOG_ERR(log_file, "Elements from \"grid\" cannot be zero or negative")
            return EXIT_FAILURE;
        }
    }

    // space_steps
    if (data.contains("space_steps")) {
        if (data["space_steps"].type() != json::value_t::array) {
            LOG_ERR(log_file, "\"space_steps\" not provided as array");
            return EXIT_FAILURE;
        }

        if (data["space_steps"].size() != 2) {
            LOG_ERR(log_file, "\"space_steps\" not 2 elements long");
            return EXIT_FAILURE;
        }

        if (data["space_steps"][0] <= 0 || data["space_steps"][1] <= 0) {
            LOG_ERR(log_file, "Elements from \"space_steps\" cannot be zero or negative")
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}

int solver_semi_lagrangian(json& data, std::ofstream& log_file) {
    LOG_INFO(log_file, "Starting the semi-lagrangian solver");

    if (check_params(data, log_file)) {
        LOG_ERR(log_file, "Problem checking the input parameters")
        return EXIT_FAILURE;
    }

    // Getting base params
    const unsigned int nx = data["grid"][0], ny = data["grid"][1];
    const float dx = data["space_steps"][0], dy = data["space_steps"][1];

    scalar_field *vx = scalar_field_init("vx", nx + 1, ny, dx, dy,log_file);
    scalar_field *vy = scalar_field_init("vy", nx, ny + 1, dx, dy,log_file);
    scalar_field *p = scalar_field_init("p", nx, ny, dx, dy,log_file);
    if (!vx || !vy || !p) {
        LOG_ERR(log_file, "An error occured initializing fields.");
        return EXIT_FAILURE;
    }


    scalar_field_free(vx, log_file);
    scalar_field_free(vy, log_file);
    scalar_field_free(p, log_file);

    return EXIT_SUCCESS;
}