#ifndef POINT_CLOUD_TRANSFORM_HPP
#define POINT_CLOUD_TRANSFORM_HPP

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <string>
#include <vector>

namespace hba {

using PointType = pcl::PointXYZI;

struct PosColumns
{
  int time = 0;
  int c1 = 1;
  int c2 = 2;
  int c3 = 3;
  int roll = 4;
  int pitch = 5;
  int yaw = 6;
};

struct Pose
{
  double t = 0.0;
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R_wb = Eigen::Matrix3d::Identity();
};

class PointCloudTransformer
{
public:
  PointCloudTransformer();

  bool LoadExtrinsicFromDeviceInfo(const std::string& json_path, bool lidar_in_body = true);
  void SetExtrinsic(const Eigen::Vector3d& t_bl, const Eigen::Matrix3d& R_bl,
                    bool lidar_in_body = false);

  bool LoadGaussPos(const std::string& pos_path, const PosColumns& cols = PosColumns());
  bool LoadWgs84PosAsNED(const std::string& pos_path, const PosColumns& cols = PosColumns());

  const std::vector<Pose>& poses() const;

  bool TransformCloudByTime(const pcl::PointCloud<PointType>& in,
                            pcl::PointCloud<PointType>& out,
                            double time, bool forward = true) const;

  bool TransformCloudByIndex(const pcl::PointCloud<PointType>& in,
                             pcl::PointCloud<PointType>& out,
                             size_t index, bool forward = true) const;

  bool TransformCloudByTimes(const pcl::PointCloud<PointType>& in,
                             const std::vector<double>& times,
                             pcl::PointCloud<PointType>& out,
                             bool forward = true) const;

  bool LoadLasFile(const std::string& las_path, pcl::PointCloud<PointType>& cloud) const;
  bool LoadLasFile(const std::string& las_path, pcl::PointCloud<PointType>& cloud,
                   std::vector<double>& times) const;

private:
  Pose LookupPose(double time) const;
  static bool ReadNumericColumns(const std::string& line, std::vector<double>& values);
  static bool ExtractJsonNumber(const std::string& json, const std::string& key, double& value);
  static Eigen::Matrix3d RPYDegToMatrix(double roll_deg, double pitch_deg, double yaw_deg);
  static Eigen::Vector3d Wgs84ToEcef(double lat_deg, double lon_deg, double h);
  static Eigen::Vector3d EcefToNed(const Eigen::Vector3d& ecef,
                                   const Eigen::Vector3d& ecef0,
                                   double lat0_deg, double lon0_deg);
  static Eigen::Vector2d Wgs84ToGauss3(double lat_deg, double lon_deg, double& cm_deg);

  Eigen::Matrix3d R_bl_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_bl_ = Eigen::Vector3d::Zero();
  bool extrinsic_ready_ = false;

  std::vector<Pose> poses_;
  bool poses_ready_ = false;
  mutable size_t last_index_ = 0;
  mutable bool last_index_ready_ = false;
};

}  // namespace hba

#endif
