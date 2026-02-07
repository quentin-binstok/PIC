#include "data.hpp"
#include "utils.hpp"
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>

scalar_field *scalar_field_init(const std::string name, const unsigned int nx,
                                const unsigned int ny, const float dx,
                                const float dy, std::ofstream &log_file) {
    LOG_INFO(log_file, "Initializing scalar field " << name);
    scalar_field *field = new scalar_field;

    field->name = name;
    field->nx = nx;
    field->ny = ny;
    field->dx = dx;
    field->dy = dy;

    field->values = (float *)calloc(nx * ny, sizeof(float));
    if (!field->values) {
        LOG_ERR(log_file, "Failed to allocate memory for scalar field "
                              << name << ", exiting.");
        return NULL;
    }

    return field;
}

void scalar_field_free(scalar_field *field, std::ofstream &log_file) {
    LOG_INFO(log_file, "Freeing scalar field " << field->name);
    free(field->values);
    free(field);
}

int write_scalar_vtk(scalar_field *data, int step, int rank,
                     std::ofstream &log_file) {
    char out[512];
    if (data->name.size() > 256) {
        LOG_ERR(log_file, "Error: data name too long for output VTK file");
        return 1;
    }
    sprintf(out, "data/%s_rank%d_%d.vti", data->name.c_str(), rank, step);

    FILE *fp = fopen(out, "wb");
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
            data->nx - 1, data->ny - 1, 0, data->dx, data->dy, 0., 0., 0., 0.,
            data->nx - 1, data->ny - 1, 0, data->name.c_str());

    fwrite(&num_bytes, sizeof(uint64_t), 1, fp);
    fwrite(data->values, sizeof(float), num_points, fp);

    fprintf(fp, "  </AppendedData>\n"
                "</VTKFile>\n");

    fclose(fp);

    return 0;
}

int write_manifest_vtk(std::string name, double dt, int nt, int sampling_rate,
                       int numranks, bool vtp, std::ofstream &log_file) {
    char out[512];
    if (name.size() > 256) {
        LOG_ERR(log_file, "Error: name too long for Paraview manifest file");
        return 1;
    }
    sprintf(out, "%s.pvd", name.c_str());

    FILE *fp = fopen(out, "wb");
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
                        const int rank, const int ndim,
                        std::ofstream &log_file) {
    char out[512];
    sprintf(out, "data/%s_rank%d_%d.vtp", field->name.c_str(), rank, step);

    FILE *fp = fopen(out, "wb");
    if (!fp)
        return 1;

    /* ---- sizes ---- */
    uint64_t bytes_points = (uint64_t)(3 * field->N * sizeof(float));
    uint64_t bytes_vel =
        field->velocity ? (uint64_t)(ndim * field->N * sizeof(float)) : 0;
    uint64_t bytes_id = field->id ? (uint64_t)(field->N * sizeof(int)) : 0;
    uint64_t bytes_conn = (uint64_t)(field->N * sizeof(int));
    uint64_t bytes_off = (uint64_t)(field->N * sizeof(int));

    /* ---- offsets into appended section ---- */
    uint64_t off_points = 0;
    uint64_t off_vel = off_points + sizeof(uint64_t) + bytes_points;
    uint64_t off_id =
        off_vel + (field->velocity ? sizeof(uint64_t) + bytes_vel : 0);
    uint64_t off_conn = off_id + (field->id ? sizeof(uint64_t) + bytes_id : 0);
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

    if (field->velocity) {
        fprintf(
            fp,
            "        <DataArray type=\"Float32\" Name=\"velocity\" "
            "NumberOfComponents=\"%d\" format=\"appended\" offset=\"%llu\"/>\n",
            ndim, (unsigned long long)off_vel);
    }

    if (field->id) {
        fprintf(fp,
                "        <DataArray type=\"Int32\" Name=\"id\" "
                "format=\"appended\" offset=\"%llu\"/>\n",
                (unsigned long long)off_id);
    }

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
        fwrite(field->xyz, sizeof(float), ndim * field->N, fp);
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
    if (field->velocity) {
        fwrite(&bytes_vel, sizeof(uint64_t), 1, fp);
        fwrite(field->velocity, sizeof(float), ndim * field->N, fp);
    }

    /* id */
    if (field->id) {
        fwrite(&bytes_id, sizeof(uint64_t), 1, fp);
        fwrite(field->id, sizeof(int), field->N, fp);
    }

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
