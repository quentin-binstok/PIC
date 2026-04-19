#include "data.hpp"
#include "utils.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

/*
 @brief Initialises a scalar field
 @param name: the name, useful when writing files
 @param nx, ny: the extend of the grid
 @param x_internal, y_internal: the position of the field inside a single cell
 @param dx: the step
 @param log_file: the log file
*/
scalar_field *scalar_field_init(const std::string name, const unsigned int nx,
                                const unsigned int ny, const float x_internal,
                                const float y_internal, const float dx,
                                std::ofstream &log_file) {
    LOG_INFO(log_file, "Initializing scalar field " << name);
    scalar_field *field = new scalar_field;
    field->name = name;
    field->nx = nx;
    field->ny = ny;
    field->x_internal = x_internal;
    field->y_internal = y_internal;
    field->dx = dx;
    // Initialises the array to zero
    field->values = (float *)calloc(nx * ny, sizeof(float));
    if (!field->values) {
        LOG_ERR(log_file, "Failed to allocate memory for scalar field "
                              << name << ", exiting.");
        return NULL;
    }
    return field;
}

/*
 @brief copies the field, but not the values (not a deep copy!)
*/
scalar_field *scalar_field_copy(const scalar_field *field,
                                std::ofstream &log_file) {
    LOG_INFO(log_file, "Copying " << field->name);
    scalar_field *new_field = new scalar_field;
    new_field->name = field->name;
    new_field->nx = field->nx;
    new_field->ny = field->ny;
    new_field->x_internal = field->x_internal;
    new_field->y_internal = field->y_internal;
    new_field->dx = field->dx;
    new_field->values = (float *)malloc(field->nx * field->ny * sizeof(float));
    if (!new_field->values) {
        LOG_ERR(log_file,
                "Failed to allocate memory to copy scalar field. Exiting");
        return NULL;
    }
    memcpy(new_field->values, field->values,
           field->nx * field->ny * sizeof(float));
    return new_field;
}

// Frees the field, including the given pointer
void scalar_field_free(scalar_field *field, std::ofstream &log_file) {
    if (!field)
        return;
    LOG_INFO(log_file, "Freeing scalar field " << field->name);
    if (field->values)
        free(field->values);
    delete field;
}

// Initializes the structure
particle_field *particle_field_init_2D(const std::string name, const int N,
                                       std::ofstream &log_file) {
    LOG_INFO(log_file, "Initializing particle field " << name);
    particle_field *field = new particle_field;

    field->name = name;
    field->N = N;
    field->next_id = N;
    field->xyz.resize(2 * N);
    field->velocity.resize(2 * N);
    /* field->B.resize(4 * N); // for APIC affine matrix */
    field->C.resize(4 * N);
    field->id.resize(N);
    field->T.resize(N);

    for (int i = 0; i < N; i++) {
        field->id[i] = i;
    }

    return field;
}

particle_field *copy_particle_field(const particle_field *particles,
                                    std::ofstream &log_file) {
    LOG_INFO(log_file, "Copying the particle field " << particles->name);

    particle_field *field = new particle_field;
    field->name = particles->name;
    field->N = particles->N;
    field->next_id = particles->next_id;
    field->xyz.assign(particles->xyz.begin(), particles->xyz.end());
    field->velocity.assign(particles->velocity.begin(),
                           particles->velocity.end());
    /* field->B.assign(particles->B.begin(), particles->B.end()); */
    field->C.assign(particles->C.begin(), particles->C.end());
    field->id.assign(particles->id.begin(), particles->id.end());

    return field;
}

// Frees the user fields
void user_field_free(user_fields *fields, std::ofstream &log) {
    if (!fields)
        return;

    LOG_INFO(log, "Freeing user fields");
    for (int i = 0; i < fields->nb_fields; i++)
        scalar_field_free(fields->fields[i], log);
    free(fields->fields);
}

// Write the scalar field to a paraview file
int write_scalar_vtk(scalar_field *data, int step, int rank,
                     std::ofstream &log_file, fs::path work_dir) {
    char out[512];
    if (data->name.size() > 256) {
        LOG_ERR(log_file, "Error: name too long for Paraview manifest file");
        return 1;
    }
    sprintf(out, "data/%s_rank%d_%d.vti", data->name.c_str(), rank, step);
    fs::path file_path = work_dir / out;
    FILE *fp = fopen(file_path.c_str(), "wb");
    if (!fp) {
        LOG_ERR(log_file, "Error: Could not open output VTK file " << out);
        return 1;
    }
    uint64_t num_points = data->nx * data->ny;
    uint64_t num_bytes = num_points * sizeof(float);
    fprintf(fp,
            "<?xml version=\"1.0\"?>\n"
            "<VTKFile"
            " type=\"ImageData\""
            " version=\"1.0\""
            " byte_order=\"LittleEndian\""
            " header_type=\"UInt64\""
            ">\n"
            "  <ImageData"
            " WholeExtent=\"0 %d 0 %d 0 %d\""
            " Spacing=\"%lf %lf %lf\""
            " Origin=\"%lf %lf %lf\""
            ">\n"
            "    <Piece Extent=\"0 %d 0 %d 0 %d\">\n"
            "      <PointData Scalars=\"scalar_data\">\n"
            "        <DataArray"
            " type=\"Float32\""
            " Name=\"%s\""
            " format=\"appended\""
            " offset=\"0\""
            ">\n"
            "        </DataArray>\n"
            "      </PointData>\n"
            "    </Piece>\n"
            "  </ImageData>\n"
            "  <AppendedData encoding=\"raw\">\n_",
            data->nx - 1, data->ny - 1, 0, data->dx, data->dx, 0.,
            data->x_internal * data->dx, data->y_internal * data->dx, 0.,
            data->nx - 1, data->ny - 1, 0, data->name.c_str());
    fwrite(&num_bytes, sizeof(uint64_t), 1, fp);
    fwrite(data->values, sizeof(float), num_points, fp);
    fprintf(fp, "  </AppendedData>\n"
                "</VTKFile>\n");
    fclose(fp);
    return 0;
}
// Writes the manifest file, use the vtp param to say if it's a vtk or vtp
int write_manifest_vtk(std::string name, double dt, int nt, int sampling_rate,
                       int numranks, bool vtp, std::ofstream &log_file,
                       fs::path work_dir) {
    char out[512];
    if (name.size() > 256) {
        LOG_ERR(log_file, "Error: name too long for Paraview manifest file");
        return 1;
    }
    sprintf(out, "%s.pvd", name.c_str());
    fs::path file_path = work_dir / out;
    FILE *fp = fopen(file_path.c_str(), "wb");
    if (!fp) {
        LOG_ERR(log_file,
                "Error: Could not open output VTK manifest file " << out);
        return 1;
    }
    fprintf(fp, "<VTKFile"
                " type=\"Collection\""
                " version=\"0.1\""
                " byte_order=\"LittleEndian\">\n"
                "  <Collection>\n");
    std::string extension;
    if (vtp)
        extension = "vtp";
    else
        extension = "vti";
    for (int n = 0; n < nt; n++) {
        if (sampling_rate && !(n % sampling_rate)) {
            double t = n * dt;
            for (int rank = 0; rank < numranks; rank++) {
                fprintf(fp,
                        "    <DataSet"
                        " timestep=\"%g\""
                        " part=\"%d\""
                        " file='data/%s_rank%d_%d.%s'/>\n",
                        t, rank, name.c_str(), rank, n, extension.c_str());
            }
        }
    }
    fprintf(fp, "  </Collection>\n"
                "</VTKFile>\n");
    fclose(fp);
    return 0;
}
// Made by chatgpt because making it ourselves is not interesting
int write_particles_vtp(const particle_field *field, const int step,
                        const int rank, const int ndim, std::ofstream &log_file,
                        fs::path work_dir) {
    char out[512];
    sprintf(out, "data/%s_rank%d_%d.vtp", field->name.c_str(), rank, step);
    fs::path file_path = work_dir / out;
    FILE *fp = fopen(file_path.c_str(), "wb");
    if (!fp)
        return 1;
    /* ---- sizes ---- */
    uint64_t bytes_points = (uint64_t)(3 * field->N * sizeof(float));
    uint64_t bytes_vel = (uint64_t)(ndim * field->N * sizeof(float));
    uint64_t bytes_id = (uint64_t)(field->N * sizeof(int));
    uint64_t bytes_conn = (uint64_t)(field->N * sizeof(int));
    uint64_t bytes_off = (uint64_t)(field->N * sizeof(int));
    /* ---- offsets into appended section ---- */
    uint64_t off_points = 0;
    uint64_t off_vel = off_points + sizeof(uint64_t) + bytes_points;
    uint64_t off_id = off_vel + sizeof(uint64_t) + bytes_vel;
    uint64_t off_conn = off_id + sizeof(uint64_t) + bytes_id;
    uint64_t off_off = off_conn + sizeof(uint64_t) + bytes_conn;
    /* ---- XML header ---- */
    fprintf(fp,
            "<?xml version=\"1.0\"?>\n"
            "<VTKFile type=\"PolyData\" version=\"1.0\" "
            "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
            "  <PolyData>\n"
            "    <Piece NumberOfPoints=\"%d\" NumberOfVerts=\"%d\">\n"
            "      <Points>\n"
            "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" "
            "Name=\"Points\" format=\"appended\" offset=\"%llu\"/>\n"
            "      </Points>\n"
            "      <PointData>\n",
            field->N, field->N, (unsigned long long)off_points);

    fprintf(fp,
            "        <DataArray type=\"Float32\" Name=\"velocity\" "
            "NumberOfComponents=\"%d\" format=\"appended\" offset=\"%llu\"/>\n",
            ndim, (unsigned long long)off_vel);

    fprintf(fp,
            "        <DataArray type=\"Int32\" Name=\"id\" "
            "format=\"appended\" offset=\"%llu\"/>\n",
            (unsigned long long)off_id);

    fprintf(fp,
            "      </PointData>\n"
            "      <Verts>\n"
            "        <DataArray type=\"Int32\" Name=\"connectivity\" "
            "format=\"appended\" offset=\"%llu\"/>\n"
            "        <DataArray type=\"Int32\" Name=\"offsets\" "
            "format=\"appended\" offset=\"%llu\"/>\n"
            "      </Verts>\n"
            "    </Piece>\n"
            "  </PolyData>\n"
            "  <AppendedData encoding=\"raw\">\n_",
            (unsigned long long)off_conn, (unsigned long long)off_off);
    /* ---- appended binary ---- */
    /* points */
    fwrite(&bytes_points, sizeof(uint64_t), 1, fp);
    if (ndim == 3)
        fwrite(field->xyz.data(), sizeof(float), ndim * field->N, fp);
    else if (ndim == 2) {
        float zero = 0;
        for (int i = 0; i < field->N; i++) {
            fwrite(&(field->xyz)[2 * i], sizeof(float), 2, fp);
            fwrite(&zero, sizeof(float), 1, fp);
        }
    } else {
        LOG_ERR(log_file, "vtp writing: ndim must be 2 or 3");
        fclose(fp);
        return EXIT_FAILURE;
    }
    /* velocity */
    fwrite(&bytes_vel, sizeof(uint64_t), 1, fp);
    fwrite(field->velocity.data(), sizeof(float), ndim * field->N, fp);

    /* id */
    fwrite(&bytes_id, sizeof(uint64_t), 1, fp);
    fwrite(field->id.data(), sizeof(int), field->N, fp);

    /* connectivity: 0,1,2,...,N-1 */
    fwrite(&bytes_conn, sizeof(uint64_t), 1, fp);
    for (int i = 0; i < field->N; i++)
        fwrite(&i, sizeof(int), 1, fp);
    /* offsets: 1,2,3,...,N */
    fwrite(&bytes_off, sizeof(uint64_t), 1, fp);
    for (int i = 1; i <= field->N; i++)
        fwrite(&i, sizeof(int), 1, fp);
    fprintf(fp, "\n  </AppendedData>\n"
                "</VTKFile>\n");
    fclose(fp);
    return 0;
}