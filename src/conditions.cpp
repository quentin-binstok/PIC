#include <cstdlib>
#include <fstream>

#include "data.hpp"
#include "nlohmann/json.hpp"
#include "utils.hpp"

using json = nlohmann::json;

/*
 @brief Applies some initial conditions to the scalar field
 @param scalar_field: the field to which apply the conditions
 @param data: the full input json
 @param condition_name: the name of the initial condition in the json
 @param log_file: the log file
*/

int initialize_domain(scalar_field *field, json &data,
                        std::string condition_name, std::ofstream &log_file) {
    LOG_INFO(log_file, "Applying initial domain " << condition_name << " on "
                                                     << field->name);

    // Checking that condition is an array
    if (data.contains(condition_name) && data[condition_name].type() != json::value_t::array) {
        LOG_ERR(log_file, "Condition " << condition_name << " is not an array");
        return EXIT_FAILURE;
    }
    // Checking that condition is the correct one
    if (condition_name == "ic_cell" && !data[condition_name].size()) {
        LOG_ERR(log_file, "Condition " << condition_name << " is not the correct one");
        return EXIT_FAILURE;
    }

    auto condition = data[condition_name];
    int nx = field->nx, ny = field->ny;

    // First we set the bc
    float value = 1.0;
    if (data.contains("bc")) {
        value = data["bc"];
    }
    for (int j = 0; j < ny; j++) {
        SET(field, 0, j, value);
        SET(field, nx - 1, j, value);
    }
    for (int i = 0; i < nx; i++) {
        SET(field, i, 0, value);
        SET(field, i, ny - 1, value);
    }

    for (int k = 0; k < (int)condition.size(); k++) {
        float value = condition[k]["value"];
        int start_x = condition[k]["tl"][0], start_y = condition[k]["tl"][1];
        int end_x = condition[k]["br"][0], end_y = condition[k]["br"][1];
        // Checking that we're in the grid
        if (start_x+1 < 0 || start_y+1 < 0 || end_x+1 > nx -1 ||
            end_y+1 > ny -1) {
            LOG_ERR(log_file, "Condition " << k << " in " << condition_name
                                           << " out of bounds");
            return EXIT_FAILURE;
        }
        // Adding the condition to the grid
        for (int j = start_y; j <= end_y; j++) {
            for (int i = start_x; i <= end_x; i++) {
                float val = GET(field, i+1, j+1);
                SET(field, i+1, j+1, value + val);
            }
        }
    }
    return EXIT_SUCCESS;
}


int initialize_vx(scalar_field *field, scalar_field *dom, json &data,
                            std::string condition_name,
                            std::ofstream &log_file) {
    LOG_INFO(log_file, "Applying " << condition_name << " on "
                                                     << field->name);

    // Checking that condition is an array
    if (data.contains(condition_name) && data[condition_name].type() != json::value_t::array) {
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
        
        if (start_x < 0 || start_y < 0 || end_x > nx -1 || end_y > ny -1) {
            LOG_ERR(log_file, "Condition " << k << " in " << condition_name
                                           << " out of bounds");
            return EXIT_FAILURE;
        }
        

        // Adding the condition to the grid
        #pragma omp parallel for collapse(2)
        for (int j = start_y; j < end_y; j++) {
            for (int i = start_x; i < end_x; i++) {
                if (GET(dom, i+1, j+1) == 1.0 || GET(dom, i, j+1) == 1.0) {
                    continue; // Skip solid cells
                }
                float val = GET(field, i, j);
                SET(field, i, j, value + val);
            }
        }
    }

    return EXIT_SUCCESS;
}
