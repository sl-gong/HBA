#include <laszip/laszip_api.h>
#include <iostream>
#include <limits>
#include <algorithm>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file.las>" << std::endl;
        return 1;
    }

    std::string path = argv[1];
    laszip_POINTER reader = nullptr;
    if (laszip_create(&reader)) {
        std::cerr << "Failed to create reader" << std::endl;
        return 1;
    }

    laszip_BOOL is_compressed = 1;
    if (laszip_open_reader(reader, path.c_str(), &is_compressed)) {
        std::cerr << "Failed to open reader" << std::endl;
        laszip_destroy(reader);
        return 1;
    }

    laszip_header* header = nullptr;
    if (laszip_get_header_pointer(reader, &header)) {
        std::cerr << "Failed to get header" << std::endl;
        laszip_destroy(reader);
        return 1;
    }

    std::cout << "File: " << path << std::endl;
    std::cout << "  Count: " << header->number_of_point_records << std::endl;
    std::cout << "  Min X/Y/Z: " << header->min_x << " " << header->min_y << " " << header->min_z << std::endl;
    std::cout << "  Max X/Y/Z: " << header->max_x << " " << header->max_y << " " << header->max_z << std::endl;
    std::cout << "  Scale X/Y/Z: " << header->x_scale_factor << " " << header->y_scale_factor << " " << header->z_scale_factor << std::endl;
    std::cout << "  Offset X/Y/Z: " << header->x_offset << " " << header->y_offset << " " << header->z_offset << std::endl;

    laszip_point* point = nullptr;
    if (laszip_get_point_pointer(reader, &point)) {
        std::cerr << "Failed to get point pointer" << std::endl;
        laszip_destroy(reader);
        return 1;
    }

    double min_t = std::numeric_limits<double>::infinity();
    double max_t = -std::numeric_limits<double>::infinity();
    
    // Check GPS time format
    bool has_time = (header->point_data_format == 1 || header->point_data_format >= 3);
    std::cout << "  Format: " << (int)header->point_data_format << (has_time ? " (Has Time)" : " (No Time)") << std::endl;

    for (laszip_U32 i = 0; i < header->number_of_point_records; ++i) {
        if (laszip_read_point(reader)) break;
        if (has_time) {
            double t = point->gps_time;
            if (t < min_t) min_t = t;
            if (t > max_t) max_t = t;
        }
    }

    if (has_time && min_t != std::numeric_limits<double>::infinity()) {
        std::cout.precision(15);
        std::cout << "  Time Range: " << min_t << " to " << max_t << std::endl;
    }

    laszip_close_reader(reader);
    laszip_destroy(reader);
    return 0;
}
