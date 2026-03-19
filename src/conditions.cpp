
#include "conditions.hpp"
#include "data.hpp"
#include "nlohmann/json.hpp"
#include "utils.hpp"
#include <cstddef>
#include <cstdlib>
#include <fstream>

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
                CELL_TYPE val = (CELL_TYPE)GET(dom, i, j);
                SET(dom, i, j, value + val);
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
        if (start_x + 1 < 0 || start_y + 1 < 0 || end_x + 1 > nx - 1 ||
            end_y + 1 > ny - 1) {
            LOG_ERR(log_file, "Condition " << k << " in " << condition_name
                                           << " out of bounds");
            return EXIT_FAILURE;
        }
// Adding the condition to the grid
#pragma omp parallel for collapse(2)
        for (int j = start_y; j < end_y; j++) {
            for (int i = start_x; i < end_x; i++) {
                if (GET(dom, i, j) == SOLID)
                    continue;
                // i+0.5 is assimilated to i, same with j
                if (field->name == "vx" && GET(dom, i + 1, j) == SOLID)
                    continue;
                if (field->name == "vy" && GET(dom, i, j + 1) == SOLID)
                    continue;
                float val = GET(field, i, j);
                SET(field, i, j, value + val);
            }
        }
    }
    return EXIT_SUCCESS;
}

// Initializes the speed fields
int boundary_condition(scalar_field *vx, scalar_field *vy, scalar_field *dom,
                       std::vector<float> &speed_condition, json &data,
                       std::string condition_name, std::ofstream &log_file) {
    LOG_INFO(log_file, "Applying " << condition_name << " on the boundaries");
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

    speed_condition.resize(4);
    // Easy access to the condition
    auto condition = data[condition_name];
    int nx = dom->nx;
    int ny = dom->ny;
    for (int k = 0; k < (int)condition.size(); k++) {
        const float type = condition[k]["type"];
        const int side = condition[k]["side"];
        speed_condition[k] = 0.0f;

        // speed is optional
        float speed_x = 0.0f;
        bool has_speed_x = condition[k].contains("speed_x");
        if (has_speed_x) {
            speed_x = condition[k]["speed_x"];
            speed_condition[k] = speed_x;
        }
        float speed_y = 0.0f;
        bool has_speed_y = condition[k].contains("speed_y");
        if (has_speed_y) {
            speed_y = condition[k]["speed_y"];
            speed_condition[k] = speed_y;
        }

        int smooth = 8;

        switch (side) {
        case 0: {
            // left boundary
            for (int j = 0; j < ny; j++) {
                SET(dom, 0, j, type);

                float u = speed_x;
                float v = speed_y;

                // bottom ramp
                if (j < smooth) {
                    u = speed_x * (float)j / smooth;
                    v = speed_y * (float)j / smooth;
                }

                // top ramp
                if (j > ny - smooth - 1) {
                    u = speed_x * (float)(ny - j - 1) / smooth;
                    v = speed_y * (float)(ny - j - 1) / smooth;
                }

                SET(vx, 0, j, u);
                SET(vy, 0, j, v);
            }
        } break;
        case 1: {
            // right boundary
            for (int j = 0; j < ny; j++) {
                SET(dom, nx - 1, j, type);
                float u = speed_x;
                float v = speed_y;
                // bottom ramp
                if (j < smooth) {
                    u = speed_x * (float)j / smooth;
                    v = speed_y * (float)j / smooth;
                }
                // top ramp
                if (j > ny - smooth - 1) {
                    u = speed_x * (float)(ny - j - 1) / smooth;
                    v = speed_y * (float)(ny - j - 1) / smooth;
                }
                SET(vx, nx - 1, j, u);
                SET(vy, nx - 1, j, v);
            }
        } break;
        case 2: {
            // top boundary
            for (int i = 0; i < nx; i++) {
                SET(dom, i, ny - 1, type);
                float u = speed_x;
                float v = speed_y;
                // left ramp
                if (i < smooth) {
                    u = speed_x * (float)i / smooth;
                    v = speed_y * (float)i / smooth;
                }
                // right ramp
                if (i > nx - smooth - 1) {
                    u = speed_x * (float)(nx - i - 1) / smooth;
                    v = speed_y * (float)(nx - i - 1) / smooth;
                }
                SET(vx, i, ny - 1, u);
                SET(vy, i, ny - 1, v);
            }
        } break;
        case 3: {
            // bottom boundary
            for (int i = 0; i < nx; i++) {
                SET(dom, i, 0, type);
                float u = speed_x;
                float v = speed_y;
                // left ramp
                if (i < smooth) {
                    u = speed_x * (float)i / smooth;
                    v = speed_y * (float)i / smooth;
                }
                // right ramp
                if (i > nx - smooth - 1) {
                    u = speed_x * (float)(nx - i - 1) / smooth;
                    v = speed_y * (float)(nx - i - 1) / smooth;
                }
                SET(vx, i, 0, u);
                SET(vy, i, 0, v);
            }
        } break;
        default:
            LOG_ERR(log_file, "Condition " << k << " in " << condition_name
                                           << " has invalid side");
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

// Function to initialize an arbitrary field
int initialize_field(scalar_field *field, std::string condition_name,
                     json &data, std::ofstream &log) {
    LOG_INFO(log, "Initializing " << field->name);

    if (!data.contains(condition_name)) {
        LOG_WARN(log, "Condition " << condition_name << " was not provided");
        return EXIT_FAILURE;
    } else if ((data[condition_name].type() != json::value_t::array)) {
        LOG_ERR(log, "Condition " << condition_name << " is not an array");
        return EXIT_FAILURE;
    }

    auto condition = data[condition_name];
    int nx = field->nx, ny = field->ny;

    for (int k = 0; k < (int)condition.size(); k++) {
        float value = condition[k]["value"];
        int start_x = condition[k]["tl"][0], start_y = condition[k]["tl"][1];
        int end_x = condition[k]["br"][0], end_y = condition[k]["br"][1];
        // Checking that we're in the grid
        if (start_x < 0 || start_y < 0 || end_x > nx - 1 || end_y > ny - 1) {
            LOG_ERR(log, "Condition " << k << " in " << condition_name
                                      << " out of bounds");
            return EXIT_FAILURE;
        }
        // Adding the condition to the grid
        for (int j = start_y; j <= end_y; j++) {
            for (int i = start_x; i <= end_x; i++) {
                float val = GET(field, i, j);
                SET(field, i, j, value + val);
            }
        }
    }

    return EXIT_SUCCESS;
}

// Function that handles the user-defined fields
int get_fields(user_fields *fields, json &data, std::ofstream &log) {
    LOG_INFO(log, "Getting the list of fields");

    int nx = data["grid"][0], ny = data["grid"][1];
    float dx = data["space_steps"];

    if (!data.contains("fields")) {
        LOG_WARN(log, "No arbitrary fields provided");
        return EXIT_SUCCESS;
    }

    if (data["fields"].type() != json::value_t::array) {
        LOG_ERR(log, "The \"fields\" parameter was not provided as an array");
        return EXIT_FAILURE;
    }

    LOG_INFO(log, "Initializing things");
    int nb_fields = data["fields"].size();
    LOG_INFO(log, "Number of fields: " << nb_fields);
    fields->nb_fields = nb_fields;
    fields->names.resize(nb_fields);
    fields->fields = NULL;
    fields->fields =
        (scalar_field **)malloc(sizeof(scalar_field *) * nb_fields);

    if (!fields->fields) {
        LOG_ERR(log, "Not possible to allocate memory for user fields");
        std::exit(1);
    }

    LOG_INFO(log, "Initializing the fields");
    for (int i = 0; i < nb_fields; i++) {
        fields->names[i] = (std::string)data["fields"][i];
        LOG_INFO(log, "Handling field " << data["fields"][i]);
        fields->fields[i] = scalar_field_init((std::string)data["fields"][i],
                                              nx, ny, 0, 0, dx, log);
        if (!fields->fields[i]) {
            LOG_ERR(log, "Not possible to allocate memory for user fields");
            for (int j = i - 1; j >= 0; j--) {
                scalar_field_free(fields->fields[j], log);
            }
            free(fields->fields);
        }
    }

    LOG_INFO(log, "Setting the user fields");
    for (int l = 0; l < fields->nb_fields; l++) {
        // auto cond = data[fields->names[l]];
        scalar_field *field = fields->fields[l];

        initialize_field(field, fields->names[l], data, log);

        // for (int k = 0; k < (int)cond.size(); k++) {
        //     float value = cond[k]["value"];
        //     int start_x = cond[k]["tl"][0], start_y = cond[k]["tl"][1];
        //     int end_x = cond[k]["br"][0], end_y = cond[k]["br"][1];
        //     // Checking that we're in the grid
        //     if (start_x < 0 || start_y < 0 || end_x > nx - 1 ||
        //         end_y > ny - 1) {
        //         LOG_ERR(log, "cond " << k << " in " << fields->names[l]
        //                              << " out of bounds");
        //         return EXIT_FAILURE;
        //     }
        //     // Adding the cond to the grid
        //     for (int j = start_y; j <= end_y; j++) {
        //         for (int i = start_x; i <= end_x; i++) {
        //             float val = GET(field, i, j);
        //             SET(field, i, j, value + val);
        //         }
        //     }
        // }
    }

    return EXIT_SUCCESS;
}

int create_circle(scalar_field *dom, std::string condition_name, json &data,
                  std::ofstream &log_file) {
    LOG_INFO(log_file, "Adding cylinders");

    if (!data.contains(condition_name)) {
        LOG_WARN(log_file,
                 "Condition " << condition_name << " was not provided");
        return EXIT_SUCCESS;
    } else if ((data[condition_name].type() != json::value_t::array)) {
        LOG_ERR(log_file, "Condition " << condition_name << " is not an array");
        return EXIT_FAILURE;
    }

    auto condition = data[condition_name];
    int nx = dom->nx, ny = dom->ny;

    for (int k = 0; k < (int)condition.size(); k++) {
        int x = condition[k]["center"][0], y = condition[k]["center"][1];
        int radius = condition[k]["radius"];

        for (int j = y - radius - 10; j <= y + radius + 10; j++) {
            for (int i = x - radius - 10; i <= x + radius + 10; i++) {
                float condition =
                    (i - x) * (i - x) + (j - y) * (j - y) - radius * radius;
                if (condition <= 0 && i >= 0 && i < nx && j >= 0 && j < ny)
                    SET(dom, i, j, 1);
            }
        }
    }

    return EXIT_SUCCESS;
}
