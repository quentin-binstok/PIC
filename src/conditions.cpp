
/*
TODO
- Make grid
- Parse BC
- Parse IC
- Handle default cases
*/
#include <cstdlib>
#include <fstream>
#include "data.hpp"
#include "nlohmann/json.hpp"
#include "utils.hpp"
using json = nlohmann::json;
int apply_initial_condition(scalar_field *field, json &data,
                            std::string condition_name,
                            std::ofstream &log_file) {
    LOG_INFO(log_file, "Applying initial condition " << condition_name << " on "
                                                     << field->name);
    // Checking that condition is an array
    if (data.contains(condition_name) &&
        data[condition_name].type() != json::value_t::array) {
        LOG_ERR(log_file, "Condition " << condition_name << " is not an array");
        return EXIT_FAILURE;
    }
    // Checking that condition exists
    if (!data.contains(condition_name) || !data[condition_name].size()) {
        LOG_WARN(log_file, "Condition " << condition_name << " unspecified");
        return EXIT_SUCCESS;
    }
    auto condition = data[condition_name];
    for (int i = 0; i < (int)condition.size(); i++) {
        const float value = condition[i]["value"];
        const int start_x = condition[i]["tl"][0],
                  start_y = condition[i]["tl"][1];
        const int end_x = condition[i]["br"][0], end_y = condition[i]["br"][1];
        if (start_x < 0 || start_y < 0 || end_x >= field->nx ||
            end_y >= field->ny) {
            LOG_ERR(log_file, "Condition " << i << " in " << condition_name
                                           << " out of bounds");
            return EXIT_FAILURE;
        }
        for (int j = condition[i]["tl"][1]; j < condition[i]["br"][1]; j++) {
            for (int k = condition[i]["tl"][0]; k < condition[i]["br"][0];
                 k++) {
                float val = GET(field, k, j);
                SET(field, k, j, value + val);
            }
        }
    }
    return EXIT_SUCCESS;
}