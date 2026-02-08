#ifndef POINT_CLOUD_TRANSFORM_HPP
#define POINT_CLOUD_TRANSFORM_HPP

#include <Eigen/Dense>

#define PCL_NO_PRECOMPILE
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/register_point_struct.h>

#include <string>
#include <vector>

namespace hba {

struct PointXYZID
{
  double time = 0.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  float intensity = 0.0f;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

using PointType = PointXYZID;

struct PosColumns
{
  int time = 0;
  int c1 = 1;
  int c2 = 2;
  int c3 = 3;
  int yaw = 4;
  int pitch = 5;
  int roll = 6;
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
  void SetNedOrigin(double lat_deg, double lon_deg, double h);
  void ClearNedOrigin();

  const std::vector<Pose>& poses() const;

  bool TransformCloudByTime(const pcl::PointCloud<PointType>& in,
                             pcl::PointCloud<PointType>& out,
                             bool forward = true) const;
  
  static double GetGaussCm(double lon_deg);
                           
  bool LoadLasFile(const std::string& las_path, pcl::PointCloud<PointType>& cloud) const;
  
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
  static Eigen::Vector2d Wgs84ToGauss3WithCm(double lat_deg, double lon_deg, double cm_deg);

  Eigen::Matrix3d R_bl_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_bl_ = Eigen::Vector3d::Zero();
  bool extrinsic_ready_ = false;

  std::vector<Pose> poses_;
  bool poses_ready_ = false;
  mutable size_t last_index_ = 0;
  mutable bool last_index_ready_ = false;

  bool ned_origin_ready_ = false;
  double ned_lat0_ = 0.0;
  double ned_lon0_ = 0.0;
  double ned_h0_ = 0.0;
  Eigen::Vector3d ned_ecef0_ = Eigen::Vector3d::Zero();
};

}  // namespace hba

POINT_CLOUD_REGISTER_POINT_STRUCT(hba::PointXYZID,
                                  (double, x, x)
                                  (double, y, y)
                                  (double, z, z)
                                  (float, intensity, intensity))

#endif
