#include "../include/point_cloud_transform.hpp"

#include <laszip/laszip_api.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>

#include <GeographicLib/Geocentric.hpp>
#include <GeographicLib/Constants.hpp>
#include <GeographicLib/LocalCartesian.hpp>
#include <GeographicLib/TransverseMercator.hpp>

namespace hba {

PointCloudTransformer::PointCloudTransformer() = default;

bool PointCloudTransformer::ExtractJsonNumber(const std::string& json,
                                              const std::string& key,
                                              double& value)
{
  std::regex rgx("\\\"" + key + "\\\"\\s*:\\s*([-+0-9.eE]+)");
  std::smatch match;
  if (std::regex_search(json, match, rgx) && match.size() >= 2)
  {
    value = std::stod(match[1].str());
    return true;
  }
  return false;
}

bool PointCloudTransformer::LoadExtrinsicFromDeviceInfo(const std::string& json_path,
                                                        bool lidar_in_body)
{
  std::ifstream file(json_path);
  if (!file.is_open())
    return false;

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string json = buffer.str();

  size_t lidar_block_pos = json.find("\"Lidar_in_Body\"");
  if (lidar_block_pos == std::string::npos)
    return false;

  size_t brace_start = json.find('{', lidar_block_pos);
  if (brace_start == std::string::npos)
    return false;

  int brace_count = 0;
  size_t brace_end = brace_start;
  for (; brace_end < json.size(); ++brace_end)
  {
    if (json[brace_end] == '{')
      ++brace_count;
    else if (json[brace_end] == '}')
    {
      --brace_count;
      if (brace_count == 0)
        break;
    }
  }

  if (brace_end >= json.size())
    return false;

  std::string lidar_block = json.substr(brace_start, brace_end - brace_start + 1);

  double ox = 0.0, oy = 0.0, oz = 0.0;
  double pit = 0.0, rol = 0.0, yaw = 0.0;

  bool ok = true;
  ok &= ExtractJsonNumber(lidar_block, "offset_x", ox);
  ok &= ExtractJsonNumber(lidar_block, "offset_y", oy);
  ok &= ExtractJsonNumber(lidar_block, "offset_z", oz);
  ok &= ExtractJsonNumber(lidar_block, "x_pit_deg", pit);
  ok &= ExtractJsonNumber(lidar_block, "y_rol_deg", rol);
  ok &= ExtractJsonNumber(lidar_block, "z_yaw_deg", yaw);

  if (!ok)
    return false;

  Eigen::Matrix3d R = RPYDegToMatrix(pit, rol, yaw);
  Eigen::Vector3d t(ox, oy, oz);
  SetExtrinsic(t, R, lidar_in_body);
  return true;
}

void PointCloudTransformer::SetExtrinsic(const Eigen::Vector3d& t_bl,
                                         const Eigen::Matrix3d& R_bl,
                                         bool lidar_in_body)
{
  if (lidar_in_body)
  {
    R_bl_ = R_bl;
    t_bl_ = t_bl;
  }
  else
  {
    Eigen::Matrix3d R_lb = R_bl;
    Eigen::Vector3d t_lb = t_bl;
    R_bl_ = R_lb.transpose();
    t_bl_ = -R_lb.transpose() * t_lb;
  }

  extrinsic_ready_ = true;
}

bool PointCloudTransformer::ReadNumericColumns(const std::string& line,
                                               std::vector<double>& values)
{
  values.clear();
  if (line.empty() || line[0] == '#' || line[0] == '%' || line[0] == '*')
    return false;

  std::stringstream ss(line);
  double v = 0.0;
  while (ss >> v)
  {
    values.push_back(v);
  }
  return values.size() >= 7;
}

Eigen::Matrix3d PointCloudTransformer::RPYDegToMatrix(double roll_deg,
                                                      double pitch_deg,
                                                      double yaw_deg)
{
  double roll = roll_deg * M_PI / 180.0;
  double pitch = pitch_deg * M_PI / 180.0;
  double yaw = yaw_deg * M_PI / 180.0;

  Eigen::AngleAxisd Rx(pitch, Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd Ry(roll, Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd Rz(yaw, Eigen::Vector3d::UnitZ());

  return (Rz * Ry * Rx).toRotationMatrix();
}

Eigen::Vector3d PointCloudTransformer::Wgs84ToEcef(double lat_deg,
                                                   double lon_deg,
                                                   double h)
{
  const auto& geocentric = GeographicLib::Geocentric::WGS84();
  double x = 0.0, y = 0.0, z = 0.0;
  geocentric.Forward(lat_deg, lon_deg, h, x, y, z);
  return Eigen::Vector3d(x, y, z);
}

Eigen::Vector3d PointCloudTransformer::EcefToNed(const Eigen::Vector3d& ecef,
                                                 const Eigen::Vector3d& ecef0,
                                                 double lat0_deg, double lon0_deg)
{
  const auto& geocentric = GeographicLib::Geocentric::WGS84();
  double lat = 0.0, lon = 0.0, h = 0.0;
  geocentric.Reverse(ecef.x(), ecef.y(), ecef.z(), lat, lon, h);

  double lat0 = 0.0, lon0 = 0.0, h0 = 0.0;
  geocentric.Reverse(ecef0.x(), ecef0.y(), ecef0.z(), lat0, lon0, h0);

  GeographicLib::LocalCartesian local(lat0, lon0, h0, geocentric);
  double x_e = 0.0, y_n = 0.0, z_u = 0.0;
  local.Forward(lat, lon, h, x_e, y_n, z_u);
  return Eigen::Vector3d(y_n, x_e, -z_u);
}

double PointCloudTransformer::GetGaussCm(double lon_deg)
{
  int zone = static_cast<int>((lon_deg + 1.5) / 3.0);
  return zone * 3.0;
}

Eigen::Vector2d PointCloudTransformer::Wgs84ToGauss3(double lat_deg,
                                                     double lon_deg,
                                                     double& cm_deg)
{
  cm_deg = GetGaussCm(lon_deg);
  return Wgs84ToGauss3WithCm(lat_deg, lon_deg, cm_deg);
}

Eigen::Vector2d PointCloudTransformer::Wgs84ToGauss3WithCm(double lat_deg,
                                                           double lon_deg,
                                                           double cm_deg)
{
  GeographicLib::TransverseMercator tm(GeographicLib::Constants::WGS84_a(),
                                       GeographicLib::Constants::WGS84_f(),
                                       1.0);
  double x = 0.0, y = 0.0, gamma = 0.0, k = 0.0;
  tm.Forward(cm_deg, lat_deg, lon_deg, x, y, gamma, k);

  double easting = 500000.0 + x;
  double northing = y;
  return Eigen::Vector2d(easting, northing);
}

bool PointCloudTransformer::LoadGaussPos(const std::string& pos_path,
                                         const PosColumns& cols)
{
  std::ifstream file(pos_path);
  if (!file.is_open())
    return false;

  poses_.clear();
  std::string line;
  std::vector<double> values;
  bool format_decided = false;
  bool input_is_wgs84 = false;
  bool cm_ready = false;
  double cm_deg = 0.0;

  while (std::getline(file, line))
  {
    if (!ReadNumericColumns(line, values))
      continue;

    if (static_cast<int>(values.size()) <= std::max({cols.time, cols.c1, cols.c2, cols.c3,
                                                     cols.roll, cols.pitch, cols.yaw}))
      continue;

    if (!format_decided)
    {
      double v1 = values[cols.c1];
      double v2 = values[cols.c2];
      input_is_wgs84 = (std::abs(v1) <= 90.0 && std::abs(v2) <= 180.0);
      format_decided = true;
      if (input_is_wgs84)
      {
        int zone = static_cast<int>((v2 + 1.5) / 3.0);
        cm_deg = zone * 3.0;
        cm_ready = true;
      }
    }

    Pose p;
    p.t = values[cols.time];
    if (input_is_wgs84)
    {
      double lat = values[cols.c1];
      double lon = values[cols.c2];
      double h = values[cols.c3];
      if (!cm_ready)
      {
        int zone = static_cast<int>((lon + 1.5) / 3.0);
        cm_deg = zone * 3.0;
        cm_ready = true;
      }
      Eigen::Vector2d gauss = Wgs84ToGauss3WithCm(lat, lon, cm_deg);
      p.p = Eigen::Vector3d(gauss.x(), gauss.y(), h);
    }
    else
    {
      p.p = Eigen::Vector3d(values[cols.c1], values[cols.c2], values[cols.c3]);
    }
    p.R_wb = RPYDegToMatrix(values[cols.roll], values[cols.pitch], values[cols.yaw]);
    poses_.push_back(p);
  }

  poses_ready_ = !poses_.empty();
  return poses_ready_;
}

bool PointCloudTransformer::LoadWgs84PosAsNED(const std::string& pos_path,
                                              const PosColumns& cols)
{
  std::ifstream file(pos_path);
  if (!file.is_open())
    return false;

  poses_.clear();
  std::string line;
  std::vector<double> values;
  bool origin_set = ned_origin_ready_;
  double lat0 = ned_lat0_;
  double lon0 = ned_lon0_;
  double h0 = ned_h0_;
  GeographicLib::LocalCartesian local;
  bool local_ready = false;

  while (std::getline(file, line))
  {
    if (!ReadNumericColumns(line, values))
      continue;

    if (static_cast<int>(values.size()) <= std::max({cols.time, cols.c1, cols.c2, cols.c3,
                                                     cols.roll, cols.pitch, cols.yaw}))
      continue;

    double t = values[cols.time];
    double lat = values[cols.c1];
    double lon = values[cols.c2];
    double h = values[cols.c3];

    if (!origin_set)
    {
      lat0 = lat;
      lon0 = lon;
      h0 = h;
      origin_set = true;
      ned_origin_ready_ = true;
      ned_lat0_ = lat0;
      ned_lon0_ = lon0;
      ned_h0_ = h0;
    }

    if (!local_ready)
    {
      local = GeographicLib::LocalCartesian(lat0, lon0, h0,
                                            GeographicLib::Geocentric::WGS84());
      local_ready = true;
    }

    double x_e = 0.0, y_n = 0.0, z_u = 0.0;
    local.Forward(lat, lon, h, x_e, y_n, z_u);
    Eigen::Vector3d ned(y_n, x_e, -z_u);

    Pose p;
    p.t = t;
    p.p = ned;
    p.R_wb = RPYDegToMatrix(values[cols.roll], values[cols.pitch], values[cols.yaw]);
    poses_.push_back(p);
  }

  poses_ready_ = !poses_.empty();
  return poses_ready_;
}

void PointCloudTransformer::SetNedOrigin(double lat_deg, double lon_deg, double h)
{
  ned_lat0_ = lat_deg;
  ned_lon0_ = lon_deg;
  ned_h0_ = h;
  ned_ecef0_ = Wgs84ToEcef(lat_deg, lon_deg, h);
  ned_origin_ready_ = true;
}

void PointCloudTransformer::ClearNedOrigin()
{
  ned_origin_ready_ = false;
}

const std::vector<Pose>& PointCloudTransformer::poses() const
{
  return poses_;
}

void PointCloudTransformer::SetPoses(const std::vector<Pose>& poses)
{
  poses_ = poses;
  poses_ready_ = !poses_.empty();
  last_index_ = 0;
  last_index_ready_ = false;
}

Pose PointCloudTransformer::LookupPose(double time) const
{
  if (poses_.empty())
    return Pose();

  if (time <= poses_.front().t)
  {
    last_index_ = 0;
    last_index_ready_ = true;
    return poses_.front();
  }

  if (time >= poses_.back().t)
  {
    last_index_ = poses_.size() - 1;
    last_index_ready_ = true;
    return poses_.back();
  }

  size_t idx = 0;
  if (!last_index_ready_ || last_index_ >= poses_.size())
  {
    size_t left = 0;
    size_t right = poses_.size() - 1;
    while (left + 1 < right)
    {
      size_t mid = (left + right) / 2;
      if (poses_[mid].t < time)
        left = mid;
      else
        right = mid;
    }
    idx = left;
  }
  else
  {
    idx = last_index_;
    if (time >= poses_[idx].t)
    {
      while (idx + 1 < poses_.size() && poses_[idx + 1].t <= time)
        ++idx;
    }
    else
    {
      while (idx > 0 && poses_[idx].t > time)
        --idx;
    }
  }

  if (idx + 1 >= poses_.size())
    idx = poses_.size() - 2;

  last_index_ = idx;
  last_index_ready_ = true;

  const Pose& p0 = poses_[idx];
  const Pose& p1 = poses_[idx + 1];

  double t0 = p0.t;
  double t1 = p1.t;
  double alpha = (t1 > t0) ? (time - t0) / (t1 - t0) : 0.0;
  alpha = std::clamp(alpha, 0.0, 1.0);

  Pose out;
  out.t = time;
  out.p = (1.0 - alpha) * p0.p + alpha * p1.p;

  Eigen::Quaterniond q0(p0.R_wb);
  Eigen::Quaterniond q1(p1.R_wb);
  Eigen::Quaterniond q = q0.slerp(alpha, q1);
  out.R_wb = q.normalized().toRotationMatrix();
  return out;
}

bool PointCloudTransformer::TransformCloudByTime(const pcl::PointCloud<PointType>& in,
                                                  pcl::PointCloud<PointType>& out,
                                                  bool forward) const
{
  if (!poses_ready_ || !extrinsic_ready_)
    return false;

  out.points.resize(in.points.size());

  for (size_t i = 0; i < in.points.size(); ++i)
  {
    const PointType& pt = in.points[i];
    Pose pose = LookupPose(pt.time);
    Eigen::Vector3d p_l(in.points[i].x, in.points[i].y, in.points[i].z);
    Eigen::Vector3d p_out;

    if (forward)
    {
      Eigen::Vector3d p_b = R_bl_ * p_l + t_bl_;
      p_out = pose.R_wb * p_b + pose.p;
    }
    else
    {
      Eigen::Vector3d p_b = pose.R_wb.transpose() * (p_l - pose.p);
      p_out = R_bl_.transpose() * (p_b - t_bl_);
    }
    out.points[i] = in.points[i];
    out.points[i].x = p_out.x();
    out.points[i].y = p_out.y();
    out.points[i].z = p_out.z();
  }
  return true;
}

bool PointCloudTransformer::LoadLasFile(const std::string& las_path,
                                        pcl::PointCloud<PointType>& cloud) const
{
  laszip_POINTER reader = nullptr;
  if (laszip_create(&reader))
    return false;

  laszip_BOOL is_compressed = 1;
  if (laszip_open_reader(reader, las_path.c_str(), &is_compressed))
  {
    laszip_destroy(reader);
    return false;
  }

  laszip_header* header = nullptr;
  if (laszip_get_header_pointer(reader, &header))
  {
    laszip_close_reader(reader);
    laszip_destroy(reader);
    return false;
  }

  laszip_point* point = nullptr;
  if (laszip_get_point_pointer(reader, &point))
  {
    laszip_close_reader(reader);
    laszip_destroy(reader);
    return false;
  }

  const laszip_U32 count = header->number_of_point_records;
  cloud.points.resize(count);

  const bool has_gps_time = (header->point_data_format == 1 || header->point_data_format == 3 ||
                             header->point_data_format == 4 || header->point_data_format == 5 ||
                             header->point_data_format == 6 || header->point_data_format == 7 ||
                             header->point_data_format == 8 || header->point_data_format == 9 ||
                             header->point_data_format == 10);
  if (!has_gps_time)
  {
    laszip_close_reader(reader);
    laszip_destroy(reader);
    return false;
  }

  for (laszip_U32 i = 0; i < count; ++i)
  {
    if (laszip_read_point(reader))
    {
      laszip_close_reader(reader);
      laszip_destroy(reader);
      return false;
    }

    double x = header->x_offset + header->x_scale_factor * point->X;
    double y = header->y_offset + header->y_scale_factor * point->Y;
    double z = header->z_offset + header->z_scale_factor * point->Z;

    cloud.points[i].x = x;
    cloud.points[i].y = y;
    cloud.points[i].z = z;
    cloud.points[i].intensity = static_cast<float>(point->intensity);
    cloud.points[i].time = point->gps_time;
  }

  laszip_close_reader(reader);
  laszip_destroy(reader);
  return true;
}


}  // namespace hba
