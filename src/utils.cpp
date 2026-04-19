#include <iostream>
#include "utils.hpp"
#include "conditions.hpp"
#include "data.hpp"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

void set_tangential_speeds(scalar_field *dom, scalar_field *vx,
                           scalar_field *vy, std::ofstream &log_file);

float interpolate_bilinear(float x, float y, float x1, float y1, float q11,
                           float q21, float q12, float q22, float Dx,
                           float Dy) {
    // x1,y1 is the bottom-left corner of the cell, Dx and Dy are the cell sizes
    // q11, q21, q12, q22 are the values at the corners (11 is bottom-left)
    float dx = x - x1;
    float dy = y - y1;
    float value = (1 - (dx / Dx) - (dy / Dy) + ((dx * dy) / (Dx * Dy))) * q11 +
                  (dx / Dx - ((dx * dy) / (Dx * Dy))) * q21 +
                  (dy / Dy - ((dx * dy) / (Dx * Dy))) * q12 +
                  ((dx * dy) / (Dx * Dy)) * q22;
    return value;
}

/*
 @brief checks the validity of parameters (except boundary and initial
 conditions)
 @param data: the whole data json
 @param log_file: the log file
 TODO: expand this with the new options
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
    // Time
    if (data.contains("nt")) {
        if (data["nt"].type() != json::value_t::number_unsigned) {
            LOG_ERR(log_file, "\"nt\" not provided as int");
            return EXIT_FAILURE;
        }
        if (data["nt"] <= 0) {
            LOG_ERR(log_file, "\"nt\" cannot be zero or negative");
            return EXIT_FAILURE;
        }
    }
    if (data.contains("delta_t")) {
        if (data["delta_t"].type() != json::value_t::number_float) {
            LOG_ERR(log_file, "\"delta_t\" not provided as float");
            return EXIT_FAILURE;
        }
        if (data["delta_t"] <= 0) {
            LOG_ERR(log_file, "\"delta_t\" cannot be zero or negative");
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

/*
 @brief interpolates the speed at x, y
 @params: v_y, v_y are the values to which the speed will be written
*/
int get_speed(float *v_x, float *v_y, float x, float y, scalar_field *vx,
              scalar_field *vy, std::ofstream &log_file) {
    float dx = vx->dx;
    // vx
    *v_x = 0;
    int x_1, x_2, y_1, y_2;
    x_1 = (int)(x / dx - 0.5);
    x_1 = std::max(0, x_1);
    x_1 = std::min(x_1, vx->nx - 2);
    if (x_1 > vx->nx - 1)
        LOG_WARN(log_file, "x_1 too big")
    x_2 = x_1 + 1;
    x_2 = std::min(vx->nx - 1, x_2);

    y_1 = (int)(y / dx);
    y_1 = std::max(0, y_1);
    y_1 = std::min(y_1, vx->ny - 2);
    if (y_1 > vx->ny - 1)
        LOG_WARN(log_file, "y_1 too big, y = " << y << " and y_1 = " << y_1)
    y_2 = y_1 + 1;
    y_2 = std::min(vx->ny - 1, y_2);

    float x0 = (x_1 + vx->x_internal) * dx;
    float y0 = (y_1 + vx->y_internal) * dx;

    *v_x =
        interpolate_bilinear(x, y, x0, y0, GET(vx, x_1, y_1), GET(vx, x_2, y_1),
                             GET(vx, x_1, y_2), GET(vx, x_2, y_2), dx, dx);

    // vy
    *v_y = 0;
    x_1 = (int)(x / dx);
    x_1 = std::max(0, x_1);
    x_1 = std::min(x_1, vy->nx - 2);
    if (x_1 > vy->nx - 1)
        LOG_WARN(log_file, "x_1 too big")

    x_2 = x_1 + 1;
    x_2 = std::min(vy->nx - 1, x_2);

    y_1 = (int)(y / dx - 0.5);
    y_1 = std::max(0, y_1);
    y_1 = std::min(y_1, vy->ny - 2);
    if (y_1 > vy->ny - 1)
        LOG_WARN(log_file, "y_1 too big")
    y_2 = y_1 + 1;
    y_2 = std::min(vy->ny - 1, y_2);

    x0 = (x_1 + vy->x_internal) * dx;
    y0 = (y_1 + vy->y_internal) * dx;

    *v_y =
        interpolate_bilinear(x, y, x0, y0, GET(vy, x_1, y_1), GET(vy, x_2, y_1),
                             GET(vy, x_1, y_2), GET(vy, x_2, y_2), dx, dx);

    return EXIT_SUCCESS;
}

/*
 @brief computes the divergence of the velocity field
 @param vx, vy: the velocity field
 @param div: the divergence field
 @param dx: the grid spacing
 @return the maximum divergence in the field, for logging purposes
*/
int divergence(scalar_field *vx, scalar_field *vy, scalar_field *div,
               scalar_field *dom, std::vector<float> speed_condition,
               std::ofstream &log_file) {
    LOG_INFO(log_file, "Computing the divergence");
    int nx = div->nx;
    int ny = div->ny;
    float dx = div->dx;

#pragma omp parallel for collapse(2)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {

            float dudx = 0.0f;
            float dvdy = 0.0f;
            if (i != 0)
                dudx = (GET(vx, i, j) - GET(vx, i - 1, j)) / dx;
            else if (GET(dom, 0, ny / 2) == AIR)
                dudx = 0;
            else
                dudx = (GET(vx, i, j) - speed_condition[0]) / dx;
            if (j != 0)
                dvdy = (GET(vy, i, j) - GET(vy, i, j - 1)) / dx;
            else if (GET(dom, nx / 2, 0) == AIR)
                dvdy = 0;
            else
                dvdy = (GET(vy, i, j) - speed_condition[3]) / dx;

            float d = dudx + dvdy;
            SET(div, i, j, d);
        }
    }
    return EXIT_SUCCESS;
}

/*
 @brief projects the velocity field to make it divergence free
 @param p: the pressure field
 @param vx, vy: the velocity field
 @param dt: the time step
 @param rho: the density
*/
int project_velocity(scalar_field *p, scalar_field *vx, scalar_field *vy,
                     scalar_field *dom, float dx, float dt, float rho,
                     std::ofstream &log_file,
                     std::vector<float> &speed_condition) {
    LOG_INFO(log_file, "Projecting the velocity field")
    int vx_nx = vx->nx;
    int vx_ny = vx->ny;
    int vy_nx = vy->nx;
    int vy_ny = vy->ny;
#pragma omp parallel for collapse(2)
    for (int j = 0; j < vx_ny; j++) {
        for (int i = 0; i < vx_nx; i++) {
            if (GET(dom, i, j) == SOLID ||
                (i != vx_nx - 1 && GET(dom, i + 1, j) == SOLID)) {
                SET(vx, i, j, 0.0);
                continue;
            }
            if (GET(dom, i, j) == DIRICHLET) {
                SET(vx, i, j, 0);
                if (i == 0) {
                    float v = speed_condition[0];
                    SET(vx, i, j, v);
                }

                if (i == vx_nx - 1) {
                    float v = speed_condition[1];
                    SET(vx, i, j, v);
                }

                continue;
            }
            if (i == vx_nx - 1) {
                float gradp_x = (GET(p, i, j) - GET(p, i - 1, j)) / dx;
                SET(vx, i, j, GET(vx, i, j) - dt * gradp_x / rho);
                continue;
            }
            float gradp_x = (GET(p, i + 1, j) - GET(p, i, j)) / dx;
            SET(vx, i, j, GET(vx, i, j) - dt * gradp_x / rho);
        }
    }

#pragma omp parallel for collapse(2)
    for (int j = 0; j < vy_ny; j++) {
        for (int i = 0; i < vy_nx; i++) {
            if (GET(dom, i, j) == SOLID ||
                (j != vy_ny - 1 && GET(dom, i, j + 1) == SOLID)) {
                SET(vy, i, j, 0.0);
                continue;
            }
            if (GET(dom, i, j) == DIRICHLET) {
                SET(vy, i, j, 0);
                if (j == 0) {
                    float v = speed_condition[3];
                    SET(vy, i, j, v);
                }

                if (j == vy_ny - 1) {
                    float v = speed_condition[2];
                    SET(vy, i, j, v);
                }
                continue;
            }
            if (j == vy_ny - 1) {
                float gradp_y = (GET(p, i, j) - GET(p, i, j - 1)) / dx;
                SET(vy, i, j, GET(vy, i, j) - dt * gradp_y / rho);
                continue;
            }
            float gradp_y = (GET(p, i, j + 1) - GET(p, i, j)) / dx;
            SET(vy, i, j, GET(vy, i, j) - dt * gradp_y / rho);
        }
    }

    // Free-slip on left/right walls
    // for (int j = 0; j < vy_ny; j++) {
    //     if (GET(dom, 0, vy_ny / 2) == SOLID)
    //         SET(vy, 0, j, GET(vy, 1, j));
    //     if (GET(dom, vy_nx - 1, vy_ny / 2) == SOLID)
    //         SET(vy, vy_nx - 1, j, GET(vy, vy_nx - 2, j));
    // }

    // // for vx on horizontal walls
    // for (int i = 0; i < vx_nx; i++) {
    //     if (GET(dom, vx_nx / 2, 0) == SOLID)
    //         SET(vx, i, 0, GET(vx, i, 1));
    //     if (GET(dom, vx_nx / 2, vx_ny - 1) == SOLID)
    //         SET(vx, i, vx_ny - 1, GET(vx, i, vx_ny - 2));
    // }

    set_tangential_speeds(dom, vx, vy, log_file);

    return EXIT_SUCCESS;
}


float volume(scalar_field *dom, float dx) {
    int nx = dom->nx;
    int ny = dom->ny;
    float volume = 0;
#pragma omp parallel for collapse(2) reduction(+ : volume)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if (GET(dom, i, j) == LIQUID) {
                volume += dx * dx;
            }
        }
    }
    return volume;
}

float free_surface_area(scalar_field *dom, float dx) {
    int nx = dom->nx;
    int ny = dom->ny;
    float area = 0;
#pragma omp parallel for collapse(2) reduction(+ : area)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if (GET(dom, i, j) == LIQUID) {
                if (i > 0 && GET(dom, i - 1, j) == AIR)
                    area += dx;
                if (i < nx - 1 && GET(dom, i + 1, j) == AIR)
                    area += dx;
                if (j > 0 && GET(dom, i, j - 1) == AIR)
                    area += dx;
                if (j < ny - 1 && GET(dom, i, j + 1) == AIR)
                    area += dx;
            }
        }
    }
    return area;
}

float depth(scalar_field *dom, int idx, float dx) {
    int ny = dom->ny;
    int depth = 0;
    for (int j = 1; j < ny; j++) {
        if (GET(dom, idx, j) == LIQUID) {
            depth++;
        } else {
            break;
        }
    }
    return depth * dx;
}
/*
 @brief sets the tangential speeds in solids and air equal to the one in liquid
*/
void set_tangential_speeds(scalar_field *dom, scalar_field *vx,
                           scalar_field *vy, std::ofstream &log_file) {
    LOG_INFO(log_file, "Setting tangential speeds");

    int nx = dom->nx, ny = dom->ny;

#pragma omp parallel for collapse(2)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            float cell_type = GET(dom, i, j);

            if (cell_type == SOLID) {
                // Getting which cells are liquid
                bool up = false, down = false, right = false, left = false;
                if (i != 0 && GET(dom, i - 1, j) == LIQUID)
                    left = true;
                if (i != nx - 1 && GET(dom, i + 1, j) == LIQUID)
                    right = true;
                if (j != 0 && GET(dom, i, j - 1) == LIQUID)
                    down = true;
                if (j != ny - 1 && GET(dom, i, j + 1) == LIQUID)
                    up = true;

                // Computing the speeds
                float horiz_speed = 0, vert_speed = 0;
                if (up)
                    horiz_speed += GET(vx, i, j + 1);
                if (down)
                    horiz_speed += GET(vx, i, j - 1);
                if (left)
                    vert_speed += GET(vy, i - 1, j);
                if (right)
                    vert_speed += GET(vy, i + 1, j);

                // First to not divide by zero, second to enforce impermeability
                if ((up || down) && !right) {
                    // Take the mean
                    horiz_speed /= (int)up + (int)down;

                    SET(vx, i, j, horiz_speed);
                }

                if ((left || right) && !up) {
                    vert_speed /= (int)left + (int)right;
                    SET(vy, i, j, vert_speed);
                }
            }
        }
    }
}

std::vector<std::string> build_headers(const json& metric_data) {
    std::vector<std::string> headers;
    for (int k = 0; k < (int)metric_data.size(); k++) {
        std::string h = metric_data[k]["header"];
        if (h == "volume")               
            headers.push_back("volume");
        else if (h == "free_surface_area") 
            headers.push_back("free_surface_area");
        else if (h == "depth")           
            headers.push_back("depth_"+ std::to_string(metric_data[k]["idx"].get<int>()));
        else if (h == "pressure")        
            headers.push_back("pressure_"+ std::to_string(metric_data[k]["idx"][0].get<int>()) + "_" + std::to_string(metric_data[k]["idx"][1].get<int>()));
        else if (h == "vx")              
            headers.push_back("vx_"+ std::to_string(metric_data[k]["idx"][0].get<int>()) + "_" + std::to_string(metric_data[k]["idx"][1].get<int>()));
        else if (h == "vy")              
            headers.push_back("vy_"+ std::to_string(metric_data[k]["idx"][0].get<int>()) + "_" + std::to_string(metric_data[k]["idx"][1].get<int>()));
        else if (h == "div")             
            headers.push_back("div_"+ std::to_string(metric_data[k]["idx"][0].get<int>()) + "_" + std::to_string(metric_data[k]["idx"][1].get<int>()));
        else if (h == "particles_solid")
            headers.push_back("particles_solid");
        else if (h == "singularity_count")
            headers.push_back("singularity_count");
    }
    return headers;
}

Metrics initialize_metrics(Metrics m, std::ofstream& log_file) {
    LOG_INFO(log_file, "Initializing metrics");
    m.particle_in_solid = 0;
    m.singularity_count = 0;
    return m;
}

Metrics compute_metrics(scalar_field *dom, scalar_field *p, scalar_field *vx, scalar_field *vy, scalar_field *div, 
                            float dx, int step, int nt, Metrics m, const json& metric_data, std::ofstream& log_file) {

    LOG_INFO(log_file, "Computing metric ");
    if (!metric_data.is_array()) {
        LOG_WARN(log_file, "Metrics not given");
        return Metrics();
    }

    m.step = step;
    m.values.clear(); 


    for (int k = 0; k < (int)metric_data.size(); k++){
        std::string h = metric_data[k]["header"];
        if (h == "volume"){
            m.values.push_back(volume(dom, dx));
            if(metric_data[k]["print"].get<bool>() && step % (nt / 10) == 0 ){
                std::cout << "Volume: " << m.values.back() << "\n";
            }
        }
        else if (h == "depth"){
            m.values.push_back(depth(dom, metric_data[k]["idx"].get<int>(), dx));
            if(metric_data[k]["print"].get<bool>() && step % (nt / 10) == 0 ){
                std::cout << "Depth at (" << metric_data[k]["idx"][0].get<int>() <<"): " << m.values.back() << "\n";
            }
        }
        else if (h == "free_surface_area"){
            m.values.push_back(free_surface_area(dom, dx));
            if(metric_data[k]["print"].get<bool>() && step % (nt / 10) == 0 ){
                std::cout << "Free surface area: " << m.values.back() << "\n";
            }
        }
        else if (h == "pressure") {
            m.values.push_back(GET(p, metric_data[k]["idx"][0].get<int>(), metric_data[k]["idx"][1].get<int>()));
            if(metric_data[k]["print"].get<bool>() && step % (nt / 10) == 0 ){
                std::cout << "Pressure at (" << metric_data[k]["idx"][0].get<int>() << ", " << metric_data[k]["idx"][1].get<int>() << "): " << m.values.back() << "\n";
            }
        }
        else if (h == "vx") {
            m.values.push_back(GET(vx, metric_data[k]["idx"][0].get<int>(), metric_data[k]["idx"][1].get<int>()));
            if(metric_data[k]["print"].get<bool>() && step % (nt / 10) == 0 ){
                std::cout << "vx at (" << metric_data[k]["idx"][0].get<int>() << ", " << metric_data[k]["idx"][1].get<int>() << "): " << m.values.back() << "\n";
            }
        }
        else if (h == "vy") {
            m.values.push_back(GET(vy, metric_data[k]["idx"][0].get<int>(), metric_data[k]["idx"][1].get<int>()));
            if(metric_data[k]["print"].get<bool>() && step % (nt / 10) == 0 ){
                std::cout << "vy at (" << metric_data[k]["idx"][0].get<int>() << ", " << metric_data[k]["idx"][1].get<int>() << "): " << m.values.back() << "\n";
            }
        }
        else if (h == "div") {
            m.values.push_back(GET(div, metric_data[k]["idx"][0].get<int>(), metric_data[k]["idx"][1].get<int>()));
            if(metric_data[k]["print"].get<bool>() && step % (nt / 10) == 0 ){
                std::cout << "Div at (" << metric_data[k]["idx"][0].get<int>() << ", " << metric_data[k]["idx"][1].get<int>() << "): " << m.values.back() << "\n";
            }
        }
        else if (h == "particles_solid") {
            m.values.push_back(m.particle_in_solid);
            if(metric_data[k]["print"].get<bool>() && step % (nt / 10) == 0 ){
                std::cout << "Particles in solid: " << m.values.back() << "\n";
            }
        }
        else if (h == "singularity_count") {
            m.values.push_back(m.singularity_count);
            if(metric_data[k]["print"].get<bool>() && step % (nt / 10) == 0 ){
                std::cout << "Singularities: " << m.values.back() << "\n";
            }
        } 
        else {
            std::cerr << "Warning: unknown metric '" << h << "', inserting 0\n";
            m.values.push_back(0.0f);
        }
    }
    return m;
}

void write_header(std::ofstream& f, const std::vector<std::string>& headers) {
    f << "step";
    for (const auto& h : headers)
        f << "," << h;
    f << "\n";
}

void write_metrics(std::ofstream& f, const Metrics& m) {
    f << m.step;
    for (const float v : m.values)
        f << "," << std::setprecision(10) << std::scientific << v;
    f << "\n";
    f.flush();
}
