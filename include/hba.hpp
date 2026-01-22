#ifndef HBA_HPP
#define HBA_HPP

#include <thread>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>

#include "mypcl.hpp"
#include "tools.hpp"
#include "ba.hpp"

class LAYER
{
public:
  int pose_size, layer_num, max_iter, part_length, left_size, left_h_size, j_upper, tail, thread_num,
    gap_num, last_win_size, left_gap_num;
  double downsample_size, voxel_size, eigen_ratio, reject_ratio;
  
  std::string data_path;
  vector<mypcl::pose> pose_vec;
  std::vector<thread*> mthreads;
  std::vector<double> mem_costs;

  std::vector<VEC(6)> hessians;
  std::vector<pcl::PointCloud<PointType>::Ptr> pcds;

  LAYER()
  {
    pose_size = 0;
    layer_num = 1;
    max_iter = 10;
    downsample_size = 0.1;
    voxel_size = 4.0;
    eigen_ratio = 0.1;
    reject_ratio = 0.05;
    pose_vec.clear(); mthreads.clear(); pcds.clear();
    hessians.clear(); mem_costs.clear();
  }

  void init_storage(int total_layer_num_)
  {
    mthreads.resize(thread_num);
    mem_costs.resize(thread_num, 0.0); // 直接resize并初始化，而不是push_back

    pcds.resize(pose_size);
    // pose_vec已经在read_pose()或init_parameter()中初始化，不需要再次resize

    #ifdef FULL_HESS
    if(layer_num < total_layer_num_)
    {
      int hessian_size = (thread_num-1)*(WIN_SIZE-1)*WIN_SIZE/2*part_length;
      hessian_size += (WIN_SIZE-1)*WIN_SIZE/2*left_gap_num;
      if(tail > 0) hessian_size += (last_win_size-1)*last_win_size/2;
      hessians.resize(hessian_size);
      printf("hessian_size: %d\n", hessian_size);
    }
    else
    {
      int hessian_size = pose_size*(pose_size-1)/2;
      hessians.resize(hessian_size);
      printf("hessian_size: %d\n", hessian_size);
    }
    #endif
  }

  void init_parameter(int pose_size_ = 0)
  {
    if(layer_num == 1)
      pose_size = pose_vec.size();
    else
      pose_size = pose_size_;
    tail = (pose_size - WIN_SIZE) % GAP;
    gap_num = (pose_size - WIN_SIZE) / GAP;
    last_win_size = pose_size - GAP * (gap_num+1);
    part_length = ceil((gap_num+1)/double(thread_num));
    
    if(gap_num-(thread_num-1)*part_length < 0) part_length = floor((gap_num+1)/double(thread_num));

    while(part_length == 0 || (gap_num-(thread_num-1)*part_length+1)/double(part_length) > 2)
    {
      thread_num -= 1;
      part_length = ceil((gap_num+1)/double(thread_num));
      if(gap_num-(thread_num-1)*part_length < 0) part_length = floor((gap_num+1)/double(thread_num));
    }
    left_gap_num = gap_num-(thread_num-1)*part_length+1;
    
    if(tail == 0)
    {
      left_size = (gap_num-(thread_num-1)*part_length+1)*WIN_SIZE;
      left_h_size = (gap_num-(thread_num-1)*part_length)*GAP+WIN_SIZE-1;
      j_upper = gap_num-(thread_num-1)*part_length+1;
    }
    else
    {
      left_size = (gap_num-(thread_num-1)*part_length+1)*WIN_SIZE+GAP+tail;
      left_h_size = (gap_num-(thread_num-1)*part_length+1)*GAP+last_win_size-1;
      j_upper = gap_num-(thread_num-1)*part_length+2;
    }

    printf("init parameter:\n");
    printf("layer_num %d | thread_num %d | pose_size %d | max_iter %d | part_length %d | gap_num %d | last_win_size %d | "
      "left_gap_num %d | tail %d | left_size %d | left_h_size %d | j_upper %d | "
      "downsample_size %f | voxel_size %f | eigen_ratio %f | reject_ratio %f\n",
      layer_num, thread_num, pose_size, max_iter, part_length, gap_num, last_win_size,
      left_gap_num, tail, left_size, left_h_size, j_upper,
      downsample_size, voxel_size, eigen_ratio, reject_ratio);
  }
};

class HBA
{
public:
  int thread_num, total_layer_num;
  std::vector<LAYER> layers;
  std::string data_path;

  HBA(int total_layer_num_, std::string data_path_, int thread_num_)
  {
    total_layer_num = total_layer_num_;
    thread_num = thread_num_;
    
    // 确保data_path以斜杠结尾
    if (!data_path_.empty() && data_path_.back() != '/') {
      data_path = data_path_ + "/";
    } else {
      data_path = data_path_;
    }

    layers.resize(total_layer_num);
    for(int i = 0; i < total_layer_num; i++)
    {
      layers[i].layer_num = i+1;
      layers[i].thread_num = thread_num;
    }
    layers[0].data_path = data_path;
    layers[0].pose_vec = mypcl::read_pose(data_path + "pose.json");
    
    // 检查pose_vec是否为空
    if (layers[0].pose_vec.empty()) {
      std::cerr << "[ERROR] HBA constructor: No pose data loaded from " << data_path + "pose.json" << std::endl;
      // 初始化一个默认姿势，避免后续崩溃
      layers[0].pose_vec.push_back(mypcl::pose());
    }
    
    layers[0].init_parameter();
    layers[0].init_storage(total_layer_num);

    for(int i = 1; i < total_layer_num; i++)
    {
      int pose_size_ = (layers[i-1].thread_num-1)*layers[i-1].part_length;
      pose_size_ += layers[i-1].tail == 0 ? layers[i-1].left_gap_num : (layers[i-1].left_gap_num+1);
      layers[i].init_parameter(pose_size_);
      layers[i].init_storage(total_layer_num);
      
      // 确保子目录路径正确构建
      layers[i].data_path = layers[i-1].data_path + "process1/";
    }
    printf("HBA init done!\n");
  }

  void update_next_layer_state(int cur_layer_num)
  {
    for(int i = 0; i < layers[cur_layer_num].thread_num; i++)
      if(i < layers[cur_layer_num].thread_num-1)
        for(int j = 0; j < layers[cur_layer_num].part_length; j++)
        {
          int index = (i * layers[cur_layer_num].part_length + j) * GAP;
          // 确保index不越界
          if (index < layers[cur_layer_num].pose_vec.size()) {
            int next_index = i * layers[cur_layer_num].part_length + j;
            if (next_index < layers[cur_layer_num+1].pose_vec.size()) {
              layers[cur_layer_num+1].pose_vec[next_index] = layers[cur_layer_num].pose_vec[index];
            }
          } else {
            std::cout << "[WARNING] update_next_layer_state: index " << index << " out of bounds (pose_vec size: " << layers[cur_layer_num].pose_vec.size() << ")" << std::endl;
          }
        }
      else
        for(int j = 0; j < layers[cur_layer_num].j_upper; j++)
        {
          int index = (i * layers[cur_layer_num].part_length + j) * GAP;
          // 确保index不越界
          if (index < layers[cur_layer_num].pose_vec.size()) {
            int next_index = i * layers[cur_layer_num].part_length + j;
            if (next_index < layers[cur_layer_num+1].pose_vec.size()) {
              layers[cur_layer_num+1].pose_vec[next_index] = layers[cur_layer_num].pose_vec[index];
            }
          } else {
            std::cout << "[WARNING] update_next_layer_state: index " << index << " out of bounds (pose_vec size: " << layers[cur_layer_num].pose_vec.size() << ")" << std::endl;
          }
        }
  }

  void pose_graph_optimization(bool use_timestamp = false, const std::string& custom_name = "", bool generate_full_cloud = false, int pcd_name_fill_num = 0)
  {
    std::cout << "[LOG] pose_graph_optimization: started." << std::endl;
    
    // 特殊情况：如果init_pose太小，直接输出结果，不进行优化
    if (layers[0].pose_vec.size() <= 1) {
      std::cout << "[LOG] pose_graph_optimization: init_pose size <= 1, skipping optimization." << std::endl;
      
      // 直接输出轨迹
      mypcl::write_pose(layers[0].pose_vec, data_path, use_timestamp, custom_name);
      
      // 如果需要生成完整点云，直接使用原始轨迹
      if (generate_full_cloud) {
        std::cout << "[LOG] pose_graph_optimization: generating full cloud without optimization." << std::endl;
        // 实现简化的点云生成逻辑
        pcl::PointCloud<PointType>::Ptr final_cloud(new pcl::PointCloud<PointType>);
        int frame_num = layers[0].pose_vec.size();
        
        for(int i = 0; i < frame_num; i++)
        {
          pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
          mypcl::loadPCD(data_path, pcd_name_fill_num, pc, i, "pcd/");
          
          if(pc->size() > 0) {
            pcl::PointCloud<PointType> transformed_pc;
            transformed_pc.points.resize(pc->points.size());
            transformed_pc.width = pc->points.size();
            transformed_pc.height = 1;
            
            for(size_t j = 0; j < pc->points.size(); j++)
            {
              Eigen::Vector3d pt_cur(pc->points[j].x, pc->points[j].y, pc->points[j].z);
              Eigen::Vector3d pt_to = layers[0].pose_vec[i].q * pt_cur + layers[0].pose_vec[i].t;
              
              transformed_pc.points[j].x = pt_to.x();
              transformed_pc.points[j].y = pt_to.y();
              transformed_pc.points[j].z = pt_to.z();
            }
            
            *final_cloud += transformed_pc;
          }
          
          if(i % 10 == 0) {
            std::cout << "Processed frame " << i << " / " << frame_num << std::endl;
          }
        }
        
        // 保存最终点云
        std::string output_file;
        if (use_timestamp) {
          std::string timestamp = mypcl::generate_timestamp();
          if (custom_name.empty()) {
            output_file = data_path + "full_cloud_" + timestamp + ".pcd";
          } else {
            output_file = data_path + custom_name + "_full_cloud_" + timestamp + ".pcd";
          }
        } else {
          if (custom_name.empty()) {
            output_file = data_path + "full_cloud.pcd";
          } else {
            output_file = data_path + custom_name + "_full_cloud.pcd";
          }
        }
        
        pcl::io::savePCDFileBinary(output_file, *final_cloud);
        std::cout << "[LOG] pose_graph_optimization: full cloud saved to " << output_file << std::endl;
      }
      
      std::cout << "[LOG] pose_graph_optimization: finished." << std::endl;
      return;
    }
    
    // 检查layers和pose_vec的有效性
    std::cout << "[LOG] pose_graph_optimization: checking layers validity..." << std::endl;
    if (total_layer_num <= 0) {
      std::cout << "[ERROR] pose_graph_optimization: total_layer_num <= 0" << std::endl;
      return;
    }
    
    if (layers.empty()) {
      std::cout << "[ERROR] pose_graph_optimization: layers vector is empty" << std::endl;
      return;
    }
    
    if (total_layer_num-1 >= layers.size()) {
      std::cout << "[ERROR] pose_graph_optimization: invalid layer index" << std::endl;
      return;
    }
    
    std::cout << "[LOG] pose_graph_optimization: copying pose vectors..." << std::endl;
    std::vector<mypcl::pose> upper_pose, init_pose;
    
    if (!layers[total_layer_num-1].pose_vec.empty()) {
      upper_pose = layers[total_layer_num-1].pose_vec;
      std::cout << "[LOG] pose_graph_optimization: upper_pose size: " << upper_pose.size() << std::endl;
    } else {
      std::cout << "[WARNING] pose_graph_optimization: upper layer pose_vec is empty" << std::endl;
    }
    
    if (!layers[0].pose_vec.empty()) {
      init_pose = layers[0].pose_vec;
      std::cout << "[LOG] pose_graph_optimization: init_pose size: " << init_pose.size() << std::endl;
    } else {
      std::cout << "[ERROR] pose_graph_optimization: init layer pose_vec is empty" << std::endl;
      return;
    }
    
    // 注意：FULL_HESS宏被注释掉了，所以hessians向量可能为空
    // 我们将使用默认的噪声模型，而不依赖于hessians向量

    std::cout << "[LOG] pose_graph_optimization: initializing gtsam objects..." << std::endl;
    gtsam::Values initial;
    gtsam::NonlinearFactorGraph graph;
    gtsam::Vector Vector6(6);
    Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8;
    
    std::cout << "[LOG] pose_graph_optimization: creating noise models..." << std::endl;
    gtsam::noiseModel::Diagonal::shared_ptr priorModel = gtsam::noiseModel::Diagonal::Variances(Vector6);
    
    std::cout << "[LOG] pose_graph_optimization: inserting initial pose..." << std::endl;
    if (init_pose.size() > 0) {
      initial.insert(0, gtsam::Pose3(gtsam::Rot3(init_pose[0].q.toRotationMatrix()), gtsam::Point3(init_pose[0].t)));
      graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, gtsam::Pose3(gtsam::Rot3(init_pose[0].q.toRotationMatrix()),
                                                               gtsam::Point3(init_pose[0].t)), priorModel));
    } else {
      std::cout << "[ERROR] pose_graph_optimization: init_pose is empty" << std::endl;
      return;
    }
    
    // 使用默认噪声模型
    gtsam::noiseModel::Diagonal::shared_ptr defaultNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);
    
    for(uint i = 0; i < init_pose.size(); i++)
    {
      if(i > 0) initial.insert(i, gtsam::Pose3(gtsam::Rot3(init_pose[i].q.toRotationMatrix()), gtsam::Point3(init_pose[i].t)));

      // 只添加相邻帧之间的约束，不使用hessians向量
      if(i < init_pose.size() - 1)
      {
        Eigen::Vector3d t_ab = init_pose[i].t;
        Eigen::Matrix3d R_ab = init_pose[i].q.toRotationMatrix();
        t_ab = R_ab.transpose() * (init_pose[i+1].t - t_ab);
        R_ab = R_ab.transpose() * init_pose[i+1].q.toRotationMatrix();
        gtsam::Rot3 R_sam(R_ab);
        gtsam::Point3 t_sam(t_ab);
        
        gtsam::NonlinearFactor::shared_ptr factor(new gtsam::BetweenFactor<gtsam::Pose3>(i, i+1, gtsam::Pose3(R_sam, t_sam),
                                                  defaultNoise));
        graph.push_back(factor);
      }
    }

    // 上层约束也使用默认噪声模型
    int pose_size = upper_pose.size();
    for(int i = 0; i < pose_size-1; i++)
    {
      Eigen::Vector3d t_ab = upper_pose[i].t;
      Eigen::Matrix3d R_ab = upper_pose[i].q.toRotationMatrix();
      t_ab = R_ab.transpose() * (upper_pose[i+1].t - t_ab);
      R_ab = R_ab.transpose() * upper_pose[i+1].q.toRotationMatrix();
      gtsam::Rot3 R_sam(R_ab);
      gtsam::Point3 t_sam(t_ab);

      gtsam::NonlinearFactor::shared_ptr factor(new gtsam::BetweenFactor<gtsam::Pose3>(i*pow(GAP, total_layer_num-1),
                                                  (i+1)*pow(GAP, total_layer_num-1), gtsam::Pose3(R_sam, t_sam), defaultNoise));
      graph.push_back(factor);
    }

    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    gtsam::ISAM2 isam(parameters);
    isam.update(graph, initial);
    isam.update();

    gtsam::Values results = isam.calculateEstimate();

    cout << "vertex size " << results.size() << endl;

    for(uint i = 0; i < results.size(); i++)
    {
      if (i < init_pose.size()) { // 确保不越界访问
        gtsam::Pose3 pose = results.at(i).cast<gtsam::Pose3>();
        assign_qt(init_pose[i].q, init_pose[i].t, Eigen::Quaterniond(pose.rotation().matrix()), pose.translation());
      }
    }
    
    // 输出带时间戳的优化轨迹
    mypcl::write_pose(init_pose, data_path, use_timestamp, custom_name);
    
    // 如果需要生成完整点云
    if (generate_full_cloud)
    {
      std::cout << "====================" << std::endl;
      std::cout << "Generating Full Point Cloud" << std::endl;
      std::cout << "====================" << std::endl;
      
      // 创建输出点云
      pcl::PointCloud<PointType>::Ptr final_cloud(new pcl::PointCloud<PointType>);
      
      // 逐帧加载点云并转换到全局坐标系
      std::cout << "Processing point clouds..." << std::endl;
      
      int frame_num = init_pose.size();
      
      for(int i = 0; i < frame_num; i++)
      {
        // 加载点云
        pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
        mypcl::loadPCD(data_path, pcd_name_fill_num, pc, i, "pcd/");
        
        if(pc->size() == 0) {
          continue;
        }
        
        // 转换点云到全局坐标系
        pcl::PointCloud<PointType> transformed_pc;
        transformed_pc.points.resize(pc->points.size());
        transformed_pc.width = pc->points.size();
        transformed_pc.height = 1;
        
        for(size_t j = 0; j < pc->points.size(); j++)
        {
          Eigen::Vector3d pt_cur(pc->points[j].x, pc->points[j].y, pc->points[j].z);
          Eigen::Vector3d pt_to = init_pose[i].q * pt_cur + init_pose[i].t;
          
          transformed_pc.points[j].x = pt_to.x();
          transformed_pc.points[j].y = pt_to.y();
          transformed_pc.points[j].z = pt_to.z();
        }
        
        // 合并到最终点云
        *final_cloud += transformed_pc;
        
        if(i % 10 == 0) {
          std::cout << "Processed frame " << i << " / " << frame_num << std::endl;
          std::cout << "Current final cloud size: " << final_cloud->size() << std::endl;
        }
      }
      
      // 体素化下采样
      std::cout << "Downsampling final point cloud..." << std::endl;
      downsample_voxel(*final_cloud, 0.05);
      
      // 保存最终点云
      std::string output_file;
      
      if (use_timestamp)
      {
        std::string timestamp = mypcl::generate_timestamp();
        if (custom_name.empty()) {
          output_file = data_path + "full_cloud_" + timestamp + ".pcd";
        } else {
          output_file = data_path + custom_name + "_full_cloud_" + timestamp + ".pcd";
        }
      } else {
        if (custom_name.empty()) {
          output_file = data_path + "full_cloud.pcd";
        } else {
          output_file = data_path + custom_name + "_full_cloud.pcd";
        }
      }
      
      std::cout << "Saving final point cloud to " << output_file << std::endl;
      pcl::io::savePCDFileBinary(output_file, *final_cloud);
      
      std::cout << "====================" << std::endl;
      std::cout << "Full Point Cloud Generated!" << std::endl;
      std::cout << "Points: " << final_cloud->size() << std::endl;
      std::cout << "Output file: " << output_file << std::endl;
      std::cout << "====================" << std::endl;
    }
    
    printf("pgo complete\n");
  }
};

#endif