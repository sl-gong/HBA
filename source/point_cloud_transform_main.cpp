#include "point_cloud_transform.hpp"

#include <pcl/io/pcd_io.h>

#include <laszip/laszip_api.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Args
{
  std::string mode;
  std::string coord;
  std::string pos_path;
  std::string device_path;
  std::string in_path;
  std::string out_path;
  bool use_time = false;
  double time = 0.0;
  bool use_index = false;
  size_t index = 0;
  bool lidar_in_body = true;
  bool use_point_time = false;
  std::string ned_origin_from;
};

void PrintUsage()
{
  std::cout << "Usage:\n"
            << "  point_cloud_transform_test \\\n    --mode forward|inverse \\\n    --coord gauss|ned \\\n    --pos /path/to/pos \\\n    --device /path/to/device_info.json \\\n    --las /path/to/input.las \\\n    --out /path/to/output.(las|pcd) \\\n    --time <timestamp> | --index <pose_index> | --use-point-time \\\n    [--lidar-in-body 0|1] [--ned-origin-from /path/to/pos]\n\n"
            << "Pos format assumptions:\n"
            << "  gauss: t x y z roll pitch yaw (deg)\n"
            << "  ned:   t lat lon h roll pitch yaw (deg)\n";
}

bool ParseArgs(int argc, char** argv, Args& args)
{
  for (int i = 1; i < argc; ++i)
  {
    std::string key(argv[i]);
    auto next = [&](std::string& out) {
      if (i + 1 >= argc)
        return false;
      out = argv[++i];
      return true;
    };

    if (key == "--mode")
    {
      if (!next(args.mode)) return false;
    }
    else if (key == "--coord")
    {
      if (!next(args.coord)) return false;
    }
    else if (key == "--pos")
    {
      if (!next(args.pos_path)) return false;
    }
    else if (key == "--device")
    {
      if (!next(args.device_path)) return false;
    }
    else if (key == "--las")
    {
      if (!next(args.in_path)) return false;
    }
    else if (key == "--out")
    {
      if (!next(args.out_path)) return false;
    }
    else if (key == "--time")
    {
      std::string v;
      if (!next(v)) return false;
      args.time = std::stod(v);
      args.use_time = true;
    }
    else if (key == "--index")
    {
      std::string v;
      if (!next(v)) return false;
      args.index = static_cast<size_t>(std::stoul(v));
      args.use_index = true;
    }
    else if (key == "--lidar-in-body")
    {
      std::string v;
      if (!next(v)) return false;
      args.lidar_in_body = (v != "0");
    }
    else if (key == "--use-point-time")
    {
      args.use_point_time = true;
    }
    else if (key == "--ned-origin-from")
    {
      if (!next(args.ned_origin_from)) return false;
    }
    else if (key == "--help" || key == "-h")
    {
      return false;
    }
  }

  if (args.mode.empty() || args.coord.empty() || args.pos_path.empty() ||
      args.device_path.empty() || args.in_path.empty() || args.out_path.empty())
    return false;

  if (!args.use_time && !args.use_index && !args.use_point_time)
    return false;

  return true;
}

bool SaveCloud(const std::string& path, const pcl::PointCloud<hba::PointType>& cloud,
               const std::vector<double>* times = nullptr)
{
  if (path.size() >= 4 && path.substr(path.size() - 4) == ".las")
  {
    laszip_POINTER writer = nullptr;
    if (laszip_create(&writer))
      return false;

    laszip_header* header = nullptr;
    if (laszip_get_header_pointer(writer, &header))
    {
      laszip_destroy(writer);
      return false;
    }

    const bool write_time = times && (times->size() == cloud.points.size());
    header->version_major = 1;
    header->version_minor = 2;
    header->point_data_format = write_time ? 1 : 0;
    header->point_data_record_length = write_time ? 28 : 20;
    header->number_of_point_records = static_cast<laszip_U32>(cloud.points.size());

    header->x_scale_factor = 0.001;
    header->y_scale_factor = 0.001;
    header->z_scale_factor = 0.001;

    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double min_z = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    double max_z = -std::numeric_limits<double>::infinity();

    for (const auto& pt : cloud.points)
    {
      min_x = std::min(min_x, static_cast<double>(pt.x));
      min_y = std::min(min_y, static_cast<double>(pt.y));
      min_z = std::min(min_z, static_cast<double>(pt.z));
      max_x = std::max(max_x, static_cast<double>(pt.x));
      max_y = std::max(max_y, static_cast<double>(pt.y));
      max_z = std::max(max_z, static_cast<double>(pt.z));
    }

    header->x_offset = std::floor(min_x);
    header->y_offset = std::floor(min_y);
    header->z_offset = std::floor(min_z);

    header->min_x = min_x;
    header->min_y = min_y;
    header->min_z = min_z;
    header->max_x = max_x;
    header->max_y = max_y;
    header->max_z = max_z;

    if (laszip_open_writer(writer, path.c_str(), 1))
    {
      laszip_destroy(writer);
      return false;
    }

    laszip_point* point = nullptr;
    if (laszip_get_point_pointer(writer, &point))
    {
      laszip_close_writer(writer);
      laszip_destroy(writer);
      return false;
    }

    for (size_t i = 0; i < cloud.points.size(); ++i)
    {
      const auto& pt = cloud.points[i];
      point->X = static_cast<laszip_I32>((pt.x - header->x_offset) / header->x_scale_factor + 0.5);
      point->Y = static_cast<laszip_I32>((pt.y - header->y_offset) / header->y_scale_factor + 0.5);
      point->Z = static_cast<laszip_I32>((pt.z - header->z_offset) / header->z_scale_factor + 0.5);
      point->intensity = static_cast<laszip_U16>(std::clamp(pt.intensity, 0.0f, 65535.0f));
      if (write_time)
        point->gps_time = (*times)[i];

      if (laszip_write_point(writer))
      {
        laszip_close_writer(writer);
        laszip_destroy(writer);
        return false;
      }
    }

    laszip_close_writer(writer);
    laszip_destroy(writer);
    return true;
  }
  return pcl::io::savePCDFileBinary(path, cloud) >= 0;
}

bool ReadFirstPosLatLonH(const std::string& path, double& lat, double& lon, double& h)
{
  std::ifstream file(path);
  if (!file.is_open())
    return false;

  std::string line;
  while (std::getline(file, line))
  {
    if (line.empty() || line[0] == '#' || line[0] == '%' || line[0] == '*')
      continue;
    std::stringstream ss(line);
    std::vector<double> values;
    double v = 0.0;
    while (ss >> v)
      values.push_back(v);
    if (values.size() >= 4)
    {
      lat = values[1];
      lon = values[2];
      h = values[3];
      return true;
    }
  }
  return false;
}

}  // namespace

int main(int argc, char** argv)
{
  Args args;
  if (!ParseArgs(argc, argv, args))
  {
    PrintUsage();
    return 1;
  }

  hba::PointCloudTransformer transformer;
  if (!transformer.LoadExtrinsicFromDeviceInfo(args.device_path, args.lidar_in_body))
  {
    std::cerr << "Failed to load device_info extrinsic." << std::endl;
    return 1;
  }

  if (args.coord == "gauss")
  {
    if (!transformer.LoadGaussPos(args.pos_path))
    {
      std::cerr << "Failed to load gauss pos." << std::endl;
      return 1;
    }
  }
  else if (args.coord == "ned")
  {
    if (!args.ned_origin_from.empty())
    {
      double lat0 = 0.0, lon0 = 0.0, h0 = 0.0;
      if (!ReadFirstPosLatLonH(args.ned_origin_from, lat0, lon0, h0))
      {
        std::cerr << "Failed to read NED origin from pos file." << std::endl;
        return 1;
      }
      std::cout << "Using NED Origin from file: " << args.ned_origin_from << std::endl;
      std::cout << "  Lat0: " << lat0 << " Lon0: " << lon0 << " H0: " << h0 << std::endl;
      transformer.SetNedOrigin(lat0, lon0, h0);
    }
    if (!transformer.LoadWgs84PosAsNED(args.pos_path))
    {
      std::cerr << "Failed to load wgs84 pos." << std::endl;
      return 1;
    }
  }
  else
  {
    std::cerr << "Unknown coord type." << std::endl;
    return 1;
  }

  pcl::PointCloud<hba::PointType> in_cloud;
  bool loaded = false;
  std::vector<double> point_times;
  if (args.in_path.size() >= 4 && args.in_path.substr(args.in_path.size() - 4) == ".pcd")
  {
    loaded = (pcl::io::loadPCDFile(args.in_path, in_cloud) >= 0);
    if (args.use_point_time)
      loaded = false;
  }
  else
  {
    if (args.use_point_time)
      loaded = transformer.LoadLasFile(args.in_path, in_cloud, point_times);
    else
      loaded = transformer.LoadLasFile(args.in_path, in_cloud);
  }

  if (!loaded)
  {
    std::cerr << "Failed to load input cloud." << std::endl;
    return 1;
  }

  if (args.use_point_time && point_times.size() != in_cloud.points.size())
  {
    std::cerr << "Per-point timestamps not available in input LAS." << std::endl;
    return 1;
  }

  pcl::PointCloud<hba::PointType> out_cloud;
  bool ok = false;
  bool forward = (args.mode == "forward");
  if (args.use_point_time)
    ok = transformer.TransformCloudByTimes(in_cloud, point_times, out_cloud, forward);
  else if (args.use_time)
    ok = transformer.TransformCloudByTime(in_cloud, out_cloud, args.time, forward);
  else
    ok = transformer.TransformCloudByIndex(in_cloud, out_cloud, args.index, forward);

  if (!ok)
  {
    std::cerr << "Transform failed." << std::endl;
    return 1;
  }

  if (!SaveCloud(args.out_path, out_cloud, args.use_point_time ? &point_times : nullptr))
  {
    std::cerr << "Failed to save output." << std::endl;
    return 1;
  }

  std::cout << "Done." << std::endl;
  return 0;
}
