#include "conditions.hpp"
#include "data.hpp"
#include "thermal.hpp"
#include "utils.hpp"

/*
 @brief advects a single particle with the APIC scheme
 @param x, y: must contain the original position, will be overwritten
*/
void advect_single_particle(float *x, float *y, scalar_field *vx,
                            scalar_field *vy, float dt,
                            std::ofstream &log_file) {

    // Three-stage third-order RK scheme
    float init_x = *x;
    float init_y = *y;

    float k_1x, k_1y;
    get_speed(&k_1x, &k_1y, init_x, init_y, vx, vy, log_file);

    float x2 = init_x + 0.5 * dt * k_1x;
    float y2 = init_y + 0.5 * dt * k_1y;
    float k_2x, k_2y;
    get_speed(&k_2x, &k_2y, x2, y2, vx, vy, log_file);

    float x3 = init_x + 0.75 * dt * k_2x;
    float y3 = init_y + 0.75 * dt * k_2y;
    float k_3x, k_3y;
    get_speed(&k_3x, &k_3y, x3, y3, vx, vy, log_file);

    float x_new = init_x + (2.0f / 9.0f) * dt * k_1x +
                  (3.0f / 9.0f) * dt * k_2x + (4.0f / 9.0f) * dt * k_3x;
    float y_new = init_y + (2.0f / 9.0f) * dt * k_1y +
                  (3.0f / 9.0f) * dt * k_2y + (4.0f / 9.0f) * dt * k_3y;

    *x = x_new;
    *y = y_new;
}

/*
 @brief avects the particles based on their velocity, using the APIC scheme
*/
int advect(particle_field *particles, scalar_field *vx, scalar_field *vy,
           float dt, std::ofstream &log_file) {
    LOG_INFO(log_file, "Advecting particles");

#pragma omp parallel for
    for (int k = 0; k < particles->N; k++) {
        // Three-stage third-order RK scheme
        float x = particles->xyz[2 * k];
        float y = particles->xyz[2 * k + 1];

        float k_1x, k_1y;
        get_speed(&k_1x, &k_1y, x, y, vx, vy, log_file);

        float x2 = x + 0.5 * dt * k_1x;
        float y2 = y + 0.5 * dt * k_1y;
        float k_2x, k_2y;
        get_speed(&k_2x, &k_2y, x2, y2, vx, vy, log_file);

        float x3 = x + 0.75 * dt * k_2x;
        float y3 = y + 0.75 * dt * k_2y;
        float k_3x, k_3y;
        get_speed(&k_3x, &k_3y, x3, y3, vx, vy, log_file);

        float x_new = x + (2.0f / 9.0f) * dt * k_1x +
                      (3.0f / 9.0f) * dt * k_2x + (4.0f / 9.0f) * dt * k_3x;
        float y_new = y + (2.0f / 9.0f) * dt * k_1y +
                      (3.0f / 9.0f) * dt * k_2y + (4.0f / 9.0f) * dt * k_3y;

        particles->xyz[2 * k] = x_new;
        particles->xyz[2 * k + 1] = y_new;
    }

    return EXIT_SUCCESS;
}

/*
 @brief the kernel used to bring back particule fields on the grid
*/
float kernel(float r) {
    if (0 <= r && r <= 1)
        return 1 - r;
    if (-1 <= r && r <= 0)
        return 1 + r;
    return 0;
}

float kernel_grad(float r) {
    if (0<= r && r <= 1)
        return -1;
    if (-1 <= r && r <= 0)
        return 1;
    return 0; // at r = 0 (any value in [-1,1] is fine numerically)
}

// Kinematic application of gravity
void apply_gravity(particle_field *particles, float g, float dt, float beta,
                   float T0) {
#pragma omp parallel for
    for (int k = 0; k < particles->N; k++) {
        particles->velocity[2 * k + 1] -=
            (1 - beta * (particles->T[k] - T0)) * g * dt;
    }
}

/*
 @brief Initializes the particles on the grid
*/
void initialize_particles(particle_field *particles, scalar_field *dom,
                          scalar_field *vx, scalar_field *vy, scalar_field *T,
                          int density, std::ofstream &log_file) {
    // Initialization of stuff
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-0.5 * dom->dx, 0.5 * dom->dx);
    int nx = dom->nx, ny = dom->ny;
    float dx = dom->dx;

    int nb_not_fluid_cases = 0;
    int current_id = 0;

    particles->xyz.resize(2 * density * nx * ny);
    particles->velocity.resize(2 * density * nx * ny);

    // DO NOT PARALLELIZE, IT BREAKS EVERYTHING
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            float cell = GET(dom, i, j);
            // Only populate liquid cells
            if (cell != SOLID && cell != AIR) {
                for (int k = 0; k < density; k++) {
                    float x, y;

                    x = i * dx + dist(gen);
                    y = j * dx + dist(gen);
                    particles->xyz[2 * current_id] = x;
                    particles->xyz[2 * current_id + 1] = y;

                    // To have speed initialization
                    float v_x, v_y;
                    get_speed(&v_x, &v_y, x, y, vx, vy, log_file);

                    particles->velocity[2 * current_id] = v_x;
                    particles->velocity[2 * current_id + 1] = v_y;

                    float temp;
                    int x_1, x_2, y_1, y_2;
                    x_1 = (int)(x / dx);
                    x_1 = std::max(0, x_1);
                    x_1 = std::min(x_1, T->nx - 2);
                    x_2 = std::min(T->nx - 1, x_1 + 1);

                    y_1 = std::max(0, (int)(y / dx));
                    y_1 = std::min(y_1, T->ny - 2);
                    y_2 = std::min(y_1 + 1, T->ny - 1);

                    float x0 = x_1 * dx;
                    float y0 = y_1 * dx;
                    temp = interpolate_bilinear(
                        x, y, x0, y0, GET(T, x_1, y_1), GET(T, x_2, y_1),
                        GET(T, x_1, y_2), GET(T, x_2, y_2), dx, dx);

                    particles->T[current_id] = temp;

                    current_id++;
                }

            } else {
                nb_not_fluid_cases++;
            }
        }
    }

    // After the loop:
    particles->N = current_id;
    particles->next_id = current_id; // next to assign = current_id
    particles->xyz.resize(2 * current_id);
    particles->velocity.resize(2 * current_id);
    /* particles->B.resize(4*current_id); */
    /* particles->C.resize(4 * current_id); */
    particles->ix.resize(3 * current_id);
    particles->iy.resize(3 * current_id);
    particles->bx.resize(2 * current_id);
    particles->by.resize(2 * current_id);

    // Fill ids properly
    particles->id.resize(current_id);

    std::iota(particles->id.begin(), particles->id.end(), 0);

    particles->T.resize(current_id);
}

// Removes a single particle of index p (not ID!)
void remove_particle(particle_field *particles, int p) {
    int last = particles->N - 1;

    if (p != last) {
        // Swap xyz (2 components per particle)
        std::swap(particles->xyz[2 * p],     particles->xyz[2 * last]);
        std::swap(particles->xyz[2 * p + 1], particles->xyz[2 * last + 1]);

        // Swap velocity (2 components per particle)
        std::swap(particles->velocity[2 * p],     particles->velocity[2 * last]);
        std::swap(particles->velocity[2 * p + 1], particles->velocity[2 * last + 1]);

        /* // Swap B (4 components per particle)
        std::swap(particles->B[4 * p],     particles->B[4 * last]);
        std::swap(particles->B[4 * p + 1], particles->B[4 * last + 1]);
        std::swap(particles->B[4 * p + 2], particles->B[4 * last + 2]);
        std::swap(particles->B[4 * p + 3], particles->B[4 * last + 3]); */
        // Swap C (4 components per particle)
        /* std::swap(particles->C[4 * p],     particles->C[4 * last]);
        std::swap(particles->C[4 * p + 1], particles->C[4 * last + 1]);
        std::swap(particles->C[4 * p + 2], particles->C[4 * last + 2]);
        std::swap(particles->C[4 * p + 3], particles->C[4 * last + 3]); */

        std::swap(particles->ix[3 * p],     particles->ix[3 * last]);
        std::swap(particles->ix[3 * p + 1], particles->ix[3 * last + 1]);
        std::swap(particles->ix[3 * p + 2], particles->ix[3 * last + 2]);

        std::swap(particles->iy[3 * p],     particles->iy[3 * last]);
        std::swap(particles->iy[3 * p + 1], particles->iy[3 * last + 1]);
        std::swap(particles->iy[3 * p + 2], particles->iy[3 * last + 2]);

        std::swap(particles->bx[2 * p],     particles->bx[2 * last]);
        std::swap(particles->bx[2 * p + 1], particles->bx[2 * last + 1]);

        std::swap(particles->by[2 * p],     particles->by[2 * last]);
        std::swap(particles->by[2 * p + 1], particles->by[2 * last + 1]);

        std::swap(particles->id[p], particles->id[last]);
        std::swap(particles->T[p], particles->T[last]);
    }

    // Remove last N elements (all O(1))
    particles->xyz.erase(particles->xyz.end() - 2, particles->xyz.end());
    particles->velocity.erase(particles->velocity.end() - 2, particles->velocity.end());
    /* particles->B.erase(particles->B.end() - 4, particles->B.end()); */
    /* particles->C.erase(particles->C.end() - 4, particles->C.end()); */
    particles->ix.erase(particles->ix.end() - 3, particles->ix.end());
    particles->iy.erase(particles->iy.end() - 3, particles->iy.end());
    particles->bx.erase(particles->bx.end() - 2, particles->bx.end());
    particles->by.erase(particles->by.end() - 2, particles->by.end());

    particles->id.pop_back();
    particles->T.pop_back();

    particles->N--;
}

/*
 @brief computes the density and removes particles in invalid places
*/
int check_particles(particle_field *particles, scalar_field *dom, Metrics &m,
                    std::vector<int> &density, std::ofstream &log_file) {
    LOG_INFO(log_file, "Updating particles")

    int nx = dom->nx;
    int ny = dom->ny;
    float dx = dom->dx;

    std::fill(density.begin(), density.end(), 0);
    m.particle_in_solid = 0;

    // Remove invalid particles
    int p = 0;
    while (p < particles->N) {

        float x = particles->xyz[2 * p];
        float y = particles->xyz[2 * p + 1];

        // exact cell
        int i = (int)(x / dx + 0.5f);
        int j = (int)(y / dx + 0.5f);

        // Check if the particle is out of bounds
        if (i < 0 || j < 0 || i >= nx || j >= ny) {
            remove_particle(particles, p);
            continue;
        }

        // Check if the particle is in a solid cell
        float cell = GET(dom, i, j);
        if (cell == SOLID) {
            remove_particle(particles, p);
            m.particle_in_solid++;
            continue;
        }

        density[j * nx + i]++;
        p++;
    }

    return EXIT_SUCCESS;
}

void compute_coeff(particle_field *particles, scalar_field *dom, scalar_field *vx, scalar_field *vy,
    Metrics &m, int p){
    int nx = vx->nx, ny = vx->ny;
    float dx = vx->dx;

    float xp = particles->xyz[2*p];
    float yp = particles->xyz[2*p+1];

    float new_u = 0.0f, new_v = 0.0f;
    float bx0 = 0.0f, bx1 = 0.0f;
    float by0 = 0.0f, by1 = 0.0f;

    float Dx00=0, Dx01=0, Dx11=0;
    float Dy00=0, Dy01=0, Dy11=0;

    // --- X faces ---
    int i0 = (int)floor(xp/dx - 0.5f);
    int j0 = (int)floor(yp/dx);

    for (int j = j0; j <= j0+1; ++j){
        for (int i = i0; i <= i0+1; ++i) {

            if (i < 0 || j < 0 || i >= nx || j >= ny) continue;

            float xf = (i + 0.5f)*dx;
            float yf = j*dx;
            float w = kernel((xp-xf)/dx)*kernel((yp-yf)/dx);
            if (w == 0) continue;

            float ox = xf - xp;
            float oy = yf - yp;

            float vi = vx->values[j*nx+i];

            new_u += w * vi;
            bx0 += w * vi * ox;
            bx1 += w * vi * oy;

            Dx00 += w * ox*ox;
            Dx01 += w * ox*oy;
            Dx11 += w * oy*oy;
        }
    }

    // --- Y faces ---
    i0 = (int)floor(xp/dx);
    j0 = (int)floor(yp/dx - 0.5f);

    for (int j = j0; j <= j0+1; ++j){
        for (int i = i0; i <= i0+1; ++i) {

            if (i < 0 || j < 0 || i >= nx || j >= ny) continue;

            float xf = i*dx;
            float yf = (j + 0.5f)*dx;
            float w = kernel((xp-xf)/dx)*kernel((yp-yf)/dx);
            if (w == 0) continue;

            float ox = xf - xp;
            float oy = yf - yp;

            float vi = vy->values[j*nx+i];

            new_v += w * vi;
            by0 += w * vi * ox;
            by1 += w * vi * oy;

            Dy00 += w * ox*ox;
            Dy01 += w * ox*oy;
            Dy11 += w * oy*oy;
        }
    }
    particles->velocity[2*p]   = new_u;
    particles->velocity[2*p+1] = new_v;
    particles->bx[2*p]   = bx0;
    particles->bx[2*p+1] = bx1;
    particles->by[2*p]   = by0;
    particles->by[2*p+1] = by1;

    int i = std::max(0, std::min(nx - 1, (int)(xp / dx)));
    int j = std::max(0, std::min(ny - 1, (int)(yp / dx)));
    float cell_type = GET(dom, i, j);
    float left_cell = GET(dom, i - 1, j);
    float right_cell = GET(dom, i + 1, j);
    float bottom_cell = GET(dom, i, j - 1);
    float top_cell = GET(dom, i, j + 1);

    // Invert D (Eq. 6)
    float detDx = Dx00*Dx11 - Dx01*Dx01;
    float detDy = Dy00*Dy11 - Dy01*Dy01;

    if (detDx < 1e-15f) {
        particles->ix[3*p+0] = 0.0f;
        particles->ix[3*p+1] = 0.0f;
        particles->ix[3*p+2] = 0.0f;
        m.singularity_count++;
    } else if (cell_type == DIRICHLET || left_cell == DIRICHLET || right_cell == DIRICHLET || bottom_cell == DIRICHLET || top_cell == DIRICHLET){ // avoid singularity in non-liquid cells (treat as PIC)
        particles->ix[3*p   ] =  4.0f/(dx*dx);
        particles->ix[3*p + 1] =  0.0f;
        particles->ix[3*p + 2] =  4.0f/(dx*dx);
        m.dirichlet++;
    } else {
        particles->ix[3*p+0] =  Dx11/detDx;
        particles->ix[3*p+1] = -Dx01/detDx;
        particles->ix[3*p+2] =  Dx00/detDx;
    }

    if (detDy < 1e-15f) {
        particles->iy[3*p+0] = 0.0f;
        particles->iy[3*p+1] = 0.0f;
        particles->iy[3*p+2] = 0.0f;
        m.singularity_count++;
    } else if (cell_type == DIRICHLET || left_cell == DIRICHLET || right_cell == DIRICHLET || bottom_cell == DIRICHLET || top_cell == DIRICHLET){ // avoid singularity in non-liquid cells (treat as PIC)
        particles->iy[3*p   ] =  4.0f/(dx*dx);
        particles->iy[3*p + 1] =  0.0f;
        particles->iy[3*p + 2] =  4.0f/(dx*dx);
        m.dirichlet++;
    } else {
        particles->iy[3*p+0] =  Dy11/detDy;
        particles->iy[3*p+1] = -Dy01/detDy;
        particles->iy[3*p+2] =  Dy00/detDy;
    }
}

/*
 @brief fills the cell (i,j) with particles until imposed_density
*/
void fill_cell(int i, int j, particle_field *particles, scalar_field *vx,
              scalar_field *vy, scalar_field *dom, scalar_field *T,
              int imposed_density, std::vector<int> &density, float dt,
              RNG &rng, Metrics &m, std::ofstream &log_file) {
    int nx = vx->nx;
    int ny = vx->ny;
    float dx = vx->dx;

    int idx = j * nx + i;
    int cell_density = density[idx];

    int to_add = std::max(0, imposed_density - cell_density);

    for (int k = 0; k < to_add; ++k) {

        // Spawn position (jittered inside cell)
        float x = i * dx + rng.jitter(rng.gen);
        float y = j * dx + rng.jitter(rng.gen);

        // Get velocity at position
        float vx_p, vy_p;
        get_speed(&vx_p, &vy_p, x, y, vx, vy, log_file);

        // Random birth time integration
        float tau = rng.birth(rng.gen);
        float remaining = dt - tau;

        advect_single_particle(&x, &y, vx, vy, remaining, log_file);

        // Compute new cell index (faster than floor)
        int fi = (int)(x / dx + 0.5f);
        int fj = (int)(y / dx + 0.5f);

        // Bounds + solid check (merged for branch efficiency)
        if (fi < 0 || fj < 0 || fi >= nx || fj >= ny)
            continue;

        if (GET(dom, fi, fj) == SOLID)
            continue;

        float temp = interp_temp(x, y, dx, T);

        // Add particle
        particles->xyz.push_back(x);
        particles->xyz.push_back(y);

        particles->velocity.push_back(vx_p);
        particles->velocity.push_back(vy_p);

        particles->bx.push_back(0);
        particles->bx.push_back(0);
        particles->by.push_back(0);
        particles->by.push_back(0);

        particles->ix.push_back(0);
        particles->ix.push_back(0);
        particles->ix.push_back(0);
        particles->iy.push_back(0);
        particles->iy.push_back(0);
        particles->iy.push_back(0);

        compute_coeff(particles, dom, vx, vy, m, particles->N);

        particles->id.push_back(particles->next_id++);
        particles->T.push_back(temp);

        // Optional: remove if you switch to size()
        particles->N++;

        // Update density
        density[fj * nx + fi]++;
    }
}

/*
 @brief refills the domain, change cell types
*/
void refill_domain(particle_field *particles, scalar_field *dom,
                   scalar_field *vx, scalar_field *vy, scalar_field *T,
                   std::vector<int> &density, int particle_density, bool refill,
                   float creation_rate, float dt, RNG &rng, Metrics &m,
                   std::ofstream &log_file)
{
    LOG_INFO(log_file, "Refilling domain");

    int nx = dom->nx;
    int ny = dom->ny;
    float dx = dom->dx;

#pragma omp parallel for collapse(2)
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {

            int idx = j * nx + i;

            float cell_type = GET(dom, i, j);
            int cell_density = density[idx];

            if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
                float left_cell = GET(dom, i - 1, j);
                float right_cell = GET(dom, i + 1, j);
                float bottom_cell = GET(dom, i, j - 1);
                float top_cell = GET(dom, i, j + 1);
                if (refill && cell_density < particle_density) {
                    if (left_cell == AIR || right_cell == AIR || bottom_cell == AIR || top_cell == AIR) {
                        float n = dt * particle_density * GET(vx, i, j)/dx; // heuristic: more particles if flow is strong
                        int target_number = (int)n;

#pragma omp critical
                        fill_cell(i, j, particles, vx, vy, dom, T, target_number, density,
                                dt, rng, m, log_file);
                        LOG_INFO(log_file, "Refilling cell (" << i << ", " << j << ") with " << target_number << " particles");
                    }
                }
            } 

            // --- DIRICHLET: inflow ---
            if (cell_type == DIRICHLET) {

                float n = dt * creation_rate;
                int target_number = (int)n;

#pragma omp critical
                fill_cell(i, j, particles, vx, vy, dom, T, target_number, density,
                          dt, rng, m, log_file);
            }

            // --- LIQUID cells ---
            else if (cell_type == LIQUID) {

                if (refill && cell_density < particle_density) {
#pragma omp critical
                    fill_cell(i, j, particles, vx, vy, dom, T, particle_density, density,
                              dt, rng, m, log_file);
                } else if (cell_density < 1) {
                    SET(dom, i, j, AIR);
                }
            }

            // --- AIR cells (convert back to liquid if needed) ---
            else if (cell_type == AIR && i > 0 && i < nx - 1 && j > 0 &&
                     j < ny - 1) {

                if (cell_density > 0) {
                    SET(dom, i, j, LIQUID);
                }
            }
        }
    }
}

