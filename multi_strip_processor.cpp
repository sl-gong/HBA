#include "include/point_cloud_transform.hpp"
#include "include/hba.hpp"
#include "include/mypcl.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <iomanip>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

namespace fs = std::filesystem;

// Structure to hold strip-specific data
struct StripData {
  std::vector<std::string> las_files;
  std::vector<std::shared_ptr<pcl::PointCloud<hba::PointType>>> clouds;
  std::vector<std::shared_ptr<pcl::PointCloud<hba::PointType>>> transformed_clouds;
  std::vector<hba::Pose> poses;
  size_t start_index;
  size_t end_index;
};

// Function to extract strip ID from filename
std::string extract_strip_id(const std::string& filename) {
  // Simple implementation: extract first part before underscore
  size_t underscore_pos = filename.find('_');
  if (underscore_pos != std::string::npos) {
    return filename.substr(0, underscore_pos);
  }
  return "default";
}

// Function to get current timestamp string
std::string get_timestamp_string() {
  auto now = std::chrono::system_clock::now();
  auto now_c = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S");
  return ss.str();
}

int main(int argc, char** argv) {
  // Check command line arguments
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <device_info.json> <mvp_pos.pos> <las_directory>" << std::endl;
    return 1;
  }

  std::string device_info_path = argv[1];
  std::string mvp_pos_path = argv[2];
  std::string las_directory = argv[3];

  // Create PointCloudTransformer instance
  hba::PointCloudTransformer transformer;

  // Load extrinsic calibration from device info
  if (!transformer.LoadExtrinsicFromDeviceInfo(device_info_path, true)) {
    std::cerr << "Failed to load extrinsic calibration from " << device_info_path << std::endl;
    return 1;
  }
  std::cout << "Extrinsic calibration loaded successfully." << std::endl;

  // Load combined navigation trajectory
  if (!transformer.LoadWgs84PosAsNED(mvp_pos_path)) {
    std::cerr << "Failed to load navigation trajectory from " << mvp_pos_path << std::endl;
    return 1;
  }
  std::cout << "Navigation trajectory loaded successfully." << std::endl;

  // Create output directory if it doesn't exist
  fs::path output_dir = fs::path(las_directory) / "aligned";
  if (!fs::exists(output_dir)) {
    if (!fs::create_directory(output_dir)) {
      std::cerr << "Failed to create output directory: " << output_dir << std::endl;
      return 1;
    }
  }

  // Create temporary HBA input directory
  std::string timestamp = get_timestamp_string();
  fs::path hba_input_dir = fs::path(las_directory) / ("hba_input_" + timestamp);
  fs::path hba_pcd_dir;
  
  try {
    if (fs::exists(hba_input_dir)) {
      std::cout << "Removing existing HBA input directory: " << hba_input_dir << std::endl;
      fs::remove_all(hba_input_dir);
    }
    
    if (!fs::create_directory(hba_input_dir)) {
      std::cerr << "Failed to create HBA input directory: " << hba_input_dir << std::endl;
      return 1;
    }
    std::cout << "Created HBA input directory: " << hba_input_dir << std::endl;
    
    // Create pcd subdirectory for HBA
    hba_pcd_dir = hba_input_dir / "pcd";
    if (!fs::create_directory(hba_pcd_dir)) {
      std::cerr << "Failed to create HBA PCD directory: " << hba_pcd_dir << std::endl;
      // Clean up and return
      fs::remove_all(hba_input_dir);
      return 1;
    }
    std::cout << "Created HBA PCD directory: " << hba_pcd_dir << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error creating HBA directories: " << e.what() << std::endl;
    return 1;
  }

  // Process each LAS file in the directory and group by strip
  std::map<std::string, StripData> strip_map;
  std::vector<std::string> all_las_files;
  
  for (const auto& entry : fs::directory_iterator(las_directory)) {
    if (entry.path().extension() == ".las") {
      all_las_files.push_back(entry.path().string());
    }
  }

  if (all_las_files.empty()) {
    std::cerr << "No LAS files found in directory: " << las_directory << std::endl;
    return 1;
  }

  std::cout << "Found " << all_las_files.size() << " LAS files to process." << std::endl;

  // Sort LAS files by name (assuming they're named by time)
  std::sort(all_las_files.begin(), all_las_files.end());

  // Group LAS files by strip
  std::cout << "Grouping LAS files by strip..." << std::endl;
  for (const auto& las_path : all_las_files) {
    fs::path las_path_obj(las_path);
    std::string filename = las_path_obj.filename().string();
    std::string strip_id = extract_strip_id(filename);
    
    strip_map[strip_id].las_files.push_back(las_path);
  }

  std::cout << "Found " << strip_map.size() << " strips to process." << std::endl;
  for (const auto& [strip_id, strip_data] : strip_map) {
    std::cout << "Strip " << strip_id << ": " << strip_data.las_files.size() << " files" << std::endl;
  }

  // First pass: process each strip and create initial transformed point clouds
  std::vector<std::shared_ptr<pcl::PointCloud<hba::PointType>>> all_clouds;
  std::vector<std::shared_ptr<pcl::PointCloud<hba::PointType>>> all_transformed_clouds;
  std::vector<std::string> all_cloud_filenames;
  size_t global_index = 0;

  // Process strips sequentially to manage memory usage
  for (auto& [strip_id, strip_data] : strip_map) {
    std::cout << "\nProcessing strip " << strip_id << "..." << std::endl;
    strip_data.start_index = global_index;
    
    for (const auto& las_path : strip_data.las_files) {
      std::cout << "Processing file " << (global_index + 1) << "/" << all_las_files.size() << ": " << las_path << std::endl;

      // Load LAS file
      auto cloud = std::make_shared<pcl::PointCloud<hba::PointType>>();
      if (!transformer.LoadLasFile(las_path, *cloud)) {
        std::cerr << "Failed to load LAS file: " << las_path << std::endl;
        continue;
      }

      std::cout << "Loaded " << cloud->size() << " points from " << las_path << std::endl;

      // Downsample point cloud to reduce memory usage
      if (cloud->size() > 1000000) {
        std::cout << "Downsampling large point cloud to reduce memory usage..." << std::endl;
        
        // Convert to PCL standard point type for downsampling
        auto pcl_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl_cloud->resize(cloud->size());
        
        for (size_t i = 0; i < cloud->size(); ++i) {
          pcl_cloud->points[i].x = cloud->points[i].x;
          pcl_cloud->points[i].y = cloud->points[i].y;
          pcl_cloud->points[i].z = cloud->points[i].z;
        }
        
        // Perform downsampling with adjusted leaf size to avoid integer overflow
        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        voxel_grid.setInputCloud(pcl_cloud);
        // Use larger leaf size for very large point clouds
        float leaf_size = cloud->size() > 5000000 ? 0.2f : 0.1f;
        voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size);
        auto downsampled_pcl_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        voxel_grid.filter(*downsampled_pcl_cloud);
        
        // Convert back to hba::PointType
        auto downsampled_cloud = std::make_shared<pcl::PointCloud<hba::PointType>>();
        downsampled_cloud->resize(downsampled_pcl_cloud->size());
        
        for (size_t i = 0; i < downsampled_pcl_cloud->size(); ++i) {
          downsampled_cloud->points[i].x = downsampled_pcl_cloud->points[i].x;
          downsampled_cloud->points[i].y = downsampled_pcl_cloud->points[i].y;
          downsampled_cloud->points[i].z = downsampled_pcl_cloud->points[i].z;
          // Copy other fields if needed
        }
        
        std::cout << "Downsampled from " << cloud->size() << " to " << downsampled_cloud->size() << " points." << std::endl;
        cloud = downsampled_cloud;
      }

      // Transform point cloud
      auto transformed_cloud = std::make_shared<pcl::PointCloud<hba::PointType>>();
      if (!transformer.TransformCloudByTime(*cloud, *transformed_cloud, true)) {
        std::cerr << "Failed to transform point cloud: " << las_path << std::endl;
        continue;
      }

      std::cout << "Transformed point cloud has " << transformed_cloud->size() << " points." << std::endl;

      // Add to strip data
      strip_data.clouds.push_back(cloud);
      strip_data.transformed_clouds.push_back(transformed_cloud);
      
      // Add to global collections
      all_clouds.push_back(cloud);
      all_transformed_clouds.push_back(transformed_cloud);
      all_cloud_filenames.push_back(las_path);
      
      global_index++;
    }
    
    strip_data.end_index = global_index - 1;
    std::cout << "Processed " << strip_data.clouds.size() << " files for strip " << strip_id << std::endl;
  }

  // Create HBA input structure
  std::cout << "\nCreating HBA input structure..." << std::endl;

  // Convert poses to mypcl::pose format
  // Use only one pose per LAS file to match the number of PCD files
  std::vector<mypcl::pose> hba_poses;
  
  // For each LAS file, select the middle pose as representative
  const auto& original_poses = transformer.poses();
  size_t total_poses = original_poses.size();
  
  if (total_poses > 0) {
    // Calculate pose indices evenly spaced across all poses
    size_t step = total_poses / all_clouds.size();
    if (step == 0) step = 1;
    
    for (size_t i = 0; i < all_clouds.size(); ++i) {
      size_t pose_index = std::min(i * step, total_poses - 1);
      Eigen::Quaterniond q(original_poses[pose_index].R_wb);
      mypcl::pose hba_pose(q, original_poses[pose_index].p);
      hba_poses.push_back(hba_pose);
    }
  }

  // Write poses to pose.json for HBA
  mypcl::write_pose(hba_poses, hba_input_dir.string() + "/");
  std::cout << "Wrote " << hba_poses.size() << " poses to pose.json (one per LAS file)" << std::endl;

  // Create PCD files for HBA (using the transformed point clouds)
  std::cout << "Creating PCD files for HBA..." << std::endl;
  
  // Process each transformed cloud
  for (size_t i = 0; i < all_transformed_clouds.size(); ++i) {
    std::string pcd_filename = hba_pcd_dir.string() + "/" + std::to_string(i) + ".pcd";
    if (pcl::io::savePCDFileBinary(pcd_filename, *all_transformed_clouds[i]) == 0) {
      std::cout << "Saved PCD file for HBA: " << pcd_filename << std::endl;
    } else {
      std::cerr << "Failed to save PCD file: " << pcd_filename << std::endl;
    }
  }
  
  // Clear memory for transformed clouds
  for (size_t i = 0; i < all_transformed_clouds.size(); ++i) {
    all_transformed_clouds[i].reset();
  }
  std::cout << "Cleared memory for transformed clouds..." << std::endl;

  // Determine optimal HBA parameters based on data size and system configuration
  int total_layer_num = 2;
  int thread_num = std::min(2, static_cast<int>(std::thread::hardware_concurrency()));
  int pcd_name_fill_num = 0;
  
  // Calculate total number of points to estimate data complexity
  size_t total_points = 0;
  for (const auto& cloud : all_clouds) {
    if (cloud) {
      total_points += cloud->size();
    }
  }
  
  std::cout << "Total points in dataset: " << total_points << std::endl;
  
  // For large datasets, use more conservative parameters to avoid memory issues
  if (total_points > 20000000) {  // 20 million points
    total_layer_num = 2;  // Use fewer layers to reduce memory usage
    thread_num = 2;  // Use fewer threads to reduce system load
    std::cout << "Adjusted HBA parameters for large dataset: layers=" << total_layer_num << ", threads=" << thread_num << std::endl;
  }
  
  // Ensure minimum thread count
  if (thread_num < 1) {
    thread_num = 1;
    std::cout << "Set minimum HBA threads to 1" << std::endl;
  }
  
  std::cout << "Final HBA parameters: layers=" << total_layer_num << ", threads=" << thread_num << ", files=" << all_las_files.size() << ", strips=" << strip_map.size() << std::endl;

  // Execute HBA
  std::cout << "\nExecuting HBA for trajectory optimization..." << std::endl;
  std::cout << "HBA parameters: layers=" << total_layer_num << ", threads=" << thread_num << std::endl;
  
  std::string hba_command = "./bin/hba " + std::to_string(total_layer_num) + " " + 
                            std::to_string(pcd_name_fill_num) + " " + 
                            hba_input_dir.string() + " " + 
                            std::to_string(thread_num);
  
  std::cout << "Executing HBA command: " << hba_command << std::endl;
  
  try {
    int hba_result = system(hba_command.c_str());
    
    if (hba_result != 0) {
      std::cerr << "HBA execution failed with return code: " << hba_result << std::endl;
      // Check if HBA binary exists
      if (!fs::exists("./bin/hba")) {
        std::cerr << "Error: HBA binary not found at ./bin/hba" << std::endl;
        std::cerr << "Please build the HBA project first using CMake." << std::endl;
      }
      // Clean up and return
      fs::remove_all(hba_input_dir);
      return 1;
    }
    std::cout << "HBA execution completed successfully!" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error executing HBA: " << e.what() << std::endl;
    // Clean up and return
    fs::remove_all(hba_input_dir);
    return 1;
  }

  // Load optimized poses from HBA
  std::cout << "\nLoading optimized poses from HBA..." << std::endl;
  std::string optimized_pose_file = hba_input_dir.string() + "/pose.json";
  std::vector<mypcl::pose> optimized_hba_poses;
  
  try {
    if (!fs::exists(optimized_pose_file)) {
      std::cerr << "Error: Optimized pose file not found: " << optimized_pose_file << std::endl;
      // Clean up and return
      fs::remove_all(hba_input_dir);
      return 1;
    }
    
    optimized_hba_poses = mypcl::read_pose(optimized_pose_file);
    
    if (optimized_hba_poses.empty()) {
      std::cerr << "Failed to load optimized poses from HBA" << std::endl;
      // Clean up and return
      fs::remove_all(hba_input_dir);
      return 1;
    }
    std::cout << "Loaded " << optimized_hba_poses.size() << " optimized poses" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error loading optimized poses: " << e.what() << std::endl;
    // Clean up and return
    fs::remove_all(hba_input_dir);
    return 1;
  }

  // Convert optimized poses back to PointCloudTransformer::Pose format
  std::vector<hba::Pose> optimized_poses;
  
  // We have one optimized pose per LAS file, but need to map back to all original poses
  size_t total_original_poses = transformer.poses().size();
  size_t optimized_pose_count = optimized_hba_poses.size();
  
  if (optimized_pose_count > 0) {
    // Initialize optimized_poses with the same size as original_poses
    optimized_poses.resize(total_original_poses);
    
    // Get reference to original poses
    const auto& original_poses = transformer.poses();
    
    // Calculate the time intervals between LAS files
    std::vector<double> file_times;
    for (size_t i = 0; i < optimized_pose_count; ++i) {
      // Use the timestamp from the original pose corresponding to this optimized pose
      size_t step = total_original_poses / optimized_pose_count;
      if (step == 0) step = 1;
      size_t original_index = std::min(i * step, total_original_poses - 1);
      file_times.push_back(original_poses[original_index].t);
    }
    
    // For each original pose, find which optimized pose interval it falls into
    for (size_t i = 0; i < total_original_poses; ++i) {
      double current_time = original_poses[i].t;
      
      // Find the nearest optimized pose
      size_t nearest_pose_idx = 0;
      double min_time_diff = std::abs(current_time - file_times[0]);
      
      for (size_t j = 1; j < optimized_pose_count; ++j) {
        double time_diff = std::abs(current_time - file_times[j]);
        if (time_diff < min_time_diff) {
          min_time_diff = time_diff;
          nearest_pose_idx = j;
        }
      }
      
      // Use the nearest optimized pose
      const auto& hba_pose = optimized_hba_poses[nearest_pose_idx];
      optimized_poses[i].p = hba_pose.t;
      optimized_poses[i].R_wb = hba_pose.q.toRotationMatrix();
      optimized_poses[i].t = original_poses[i].t;  // Preserve original timestamp
    }
  } else {
    std::cerr << "Error: No optimized poses available!" << std::endl;
    // Clean up and return
    fs::remove_all(hba_input_dir);
    return 1;
  }

  std::cout << "Mapped " << optimized_hba_poses.size() << " optimized poses back to " 
            << optimized_poses.size() << " original poses" << std::endl;

  // Set optimized poses in transformer
  transformer.SetPoses(optimized_poses);
  std::cout << "Set optimized poses in PointCloudTransformer" << std::endl;

  // Second pass: re-transform point clouds with optimized poses
  std::cout << "\nRe-transforming point clouds with optimized poses..." << std::endl;
  for (size_t i = 0; i < all_clouds.size(); ++i) {
    const std::string& las_path = all_cloud_filenames[i];
    std::cout << "Re-processing file " << (i + 1) << "/" << all_clouds.size() << ": " << las_path << std::endl;

    // Re-transform point cloud with optimized poses
    auto optimized_transformed_cloud = std::make_shared<pcl::PointCloud<hba::PointType>>();
    if (!transformer.TransformCloudByTime(*all_clouds[i], *optimized_transformed_cloud, true)) {
      std::cerr << "Failed to re-transform point cloud: " << las_path << std::endl;
      continue;
    }

    std::cout << "Optimized transformed point cloud has " << optimized_transformed_cloud->size() << " points." << std::endl;

    // Save optimized transformed point cloud
    fs::path input_path(las_path);
    std::string output_filename = "aligned_" + input_path.filename().string();
    fs::path output_path = output_dir / output_filename;

    // Save as PCD file
    std::string pcd_output_path = output_path.string();
    pcd_output_path.replace(pcd_output_path.end() - 3, pcd_output_path.end(), "pcd");
    
    if (pcl::io::savePCDFileBinary(pcd_output_path, *optimized_transformed_cloud) == 0) {
      std::cout << "Saved optimized aligned point cloud to: " << pcd_output_path << std::endl;
    } else {
      std::cerr << "Failed to save optimized aligned point cloud to: " << pcd_output_path << std::endl;
    }
  }

  // Create combined point cloud for each strip
  std::cout << "\nCreating combined point clouds for each strip..." << std::endl;
  for (const auto& [strip_id, strip_data] : strip_map) {
    if (strip_data.transformed_clouds.empty()) {
      std::cerr << "No transformed clouds for strip " << strip_id << std::endl;
      continue;
    }
    
    auto combined_cloud = std::make_shared<pcl::PointCloud<hba::PointType>>();
    for (const auto& cloud : strip_data.transformed_clouds) {
      *combined_cloud += *cloud;
    }
    
    std::string combined_filename = "strip_" + strip_id + "_combined.pcd";
    fs::path combined_path = output_dir / combined_filename;
    
    if (pcl::io::savePCDFileBinary(combined_path.string(), *combined_cloud) == 0) {
      std::cout << "Saved combined point cloud for strip " << strip_id << " to: " << combined_path << std::endl;
      std::cout << "Combined cloud has " << combined_cloud->size() << " points." << std::endl;
    } else {
      std::cerr << "Failed to save combined point cloud for strip " << strip_id << std::endl;
    }
  }

  // Clean up temporary HBA input directory
  std::cout << "\nCleaning up temporary HBA input directory..." << std::endl;
  fs::remove_all(hba_input_dir);
  std::cout << "Cleaned up temporary HBA input directory" << std::endl;

  std::cout << "\nMulti-strip processing with HBA trajectory optimization completed successfully!" << std::endl;
  std::cout << "Processed " << strip_map.size() << " strips containing " << all_las_files.size() << " LAS files." << std::endl;
  return 0;
}
