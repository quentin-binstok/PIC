#include "conditions.hpp"
#include <cstdlib>
#include <fstream>

#include "data.hpp"
#include "nlohmann/json.hpp"
#include "utils.hpp"

using json = nlohmann::json;

/*
 @brief Applies some initial conditions to the scalar field
 @param scalar_field: the field to which apply the conditions (the domain)
 @param data: the full input json
 @param condition_name: the name of the initial condition in the json
 @param log_file: the log file
*/
int initialize_domain(scalar_field *dom, json &data, std::string condition_name,
                      std::ofstream &log_file) {
    LOG_INFO(log_file, "Applying initial domain " << condition_name << " on "
                                                  << dom->name);

    auto condition = data[condition_name];
    int nx = dom->nx, ny = dom->ny;

    // First we set the bc, around the domain
    CELL_TYPE value = SOLID;
    if (data.contains("bc")) {
        value = data["bc"];
    }
    for (int j = 0; j < ny; j++) {
        SET(dom, 0, j, value);
        SET(dom, nx - 1, j, value);
    }
    for (int i = 0; i < nx; i++) {
        SET(dom, i, 0, value);
        SET(dom, i, ny - 1, value);
    }

    for (int k = 0; k < (int)condition.size(); k++) {
        CELL_TYPE value = condition[k]["value"];
        int start_x = condition[k]["tl"][0], start_y = condition[k]["tl"][1];
        int end_x = condition[k]["br"][0], end_y = condition[k]["br"][1];
        // Checking that we're in the grid
        if (start_x + 1 < 0 || start_y + 1 < 0 || end_x + 1 > nx - 1 ||
            end_y + 1 > ny - 1) {
            LOG_ERR(log_file, "Condition " << k << " in " << condition_name
                                           << " out of bounds");
            return EXIT_FAILURE;
        }
        // Adding the condition to the grid
        for (int j = start_y; j <= end_y; j++) {
            for (int i = start_x; i <= end_x; i++) {
                CELL_TYPE val = (CELL_TYPE)GET(dom, i + 1, j + 1);
                SET(dom, i + 1, j + 1, value + val);
            }
        }
    }
    return EXIT_SUCCESS;
}

// Initializes the speed fields
int initialize_speed(scalar_field *field, scalar_field *dom, json &data,
                     std::string condition_name, std::ofstream &log_file) {
    LOG_INFO(log_file, "Applying " << condition_name << " on " << field->name);

    if (!data.contains(condition_name)) {
        LOG_WARN(log_file, "Condition " << condition_name << " not given");
        return EXIT_SUCCESS;
    }

    // Checking that condition is an array
    if (data.contains(condition_name) &&
        data[condition_name].type() != json::value_t::array) {
        LOG_ERR(log_file, "Condition " << condition_name << " is not an array");
        return EXIT_FAILURE;
    }
    // Easy access to the condition
    auto condition = data[condition_name];
    int nx = field->nx;
    int ny = field->ny;

    for (int k = 0; k < (int)condition.size(); k++) {
        const float value = condition[k]["value"];
        const int start_x = condition[k]["tl"][0];
        const int start_y = condition[k]["tl"][1];
        const int end_x = condition[k]["br"][0];
        const int end_y = condition[k]["br"][1];

        if (start_x < 0 || start_y < 0 || end_x > nx - 1 || end_y > ny - 1) {
            LOG_ERR(log_file, "Condition " << k << " in " << condition_name
                                           << " out of bounds");
            return EXIT_FAILURE;
        }

// Adding the condition to the grid
#pragma omp parallel for collapse(2)
        for (int j = start_y; j < end_y; j++) {
            for (int i = start_x; i < end_x; i++) {
                if (GET(dom, i + 1, j + 1) == SOLID)
                    continue;

                // i+0.5 is assimilated to i, same with j
                if (field->name == "vx" && GET(dom, i + 2, j + 1) == SOLID)
                    continue;

                if (field->name == "vy" && GET(dom, i + 1, j + 2) == SOLID)
                    continue;

                float val = GET(field, i, j);
                SET(field, i, j, value + val);
            }
        }
    }

    return EXIT_SUCCESS;
}
