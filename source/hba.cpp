#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <chrono>

#include <mutex>
#include <assert.h>
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "ba.hpp"
#include "hba.hpp"
#include "tools.hpp"
#include "mypcl.hpp"

using namespace std;
using namespace Eigen;

int pcd_name_fill_num = 0;

void cut_voxel(unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*>& feat_map,
                pcl::PointCloud<PointType>& feat_pt,
                Eigen::Quaterniond q, Eigen::Vector3d t, int fnum,
                double voxel_size, int window_size, float eigen_ratio)
{
  float loc_xyz[3];
  for(PointType& p_c: feat_pt.points)
  {
    Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
    Eigen::Vector3d pvec_tran = q * pvec_orig + t;

    for(int j = 0; j < 3; j++)
    {
      loc_xyz[j] = pvec_tran[j] / voxel_size;
      if(loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if(iter != feat_map.end())
    {
      iter->second->vec_orig[fnum].push_back(pvec_orig);
      iter->second->vec_tran[fnum].push_back(pvec_tran);

      iter->second->sig_orig[fnum].push(pvec_orig);
      iter->second->sig_tran[fnum].push(pvec_tran);
    }
    else
    {
      OCTO_TREE_ROOT* ot = new OCTO_TREE_ROOT(window_size, eigen_ratio);
      ot->vec_orig[fnum].push_back(pvec_orig);
      ot->vec_tran[fnum].push_back(pvec_tran);
      ot->sig_orig[fnum].push(pvec_orig);
      ot->sig_tran[fnum].push(pvec_tran);

      ot->voxel_center[0] = (0.5+position.x) * voxel_size;
      ot->voxel_center[1] = (0.5+position.y) * voxel_size;
      ot->voxel_center[2] = (0.5+position.z) * voxel_size;
      ot->quater_length = voxel_size / 4.0;
      ot->layer = 0;
      feat_map[position] = ot;
    }
  }
}

void parallel_comp(LAYER& layer, int thread_id, LAYER& next_layer)
{
  std::cout << "[LOG] parallel_comp: thread_id=" << thread_id << " started." << std::endl;
  
  int& part_length = layer.part_length;
  int& layer_num = layer.layer_num;
  
  std::cout << "[LOG] parallel_comp: thread_id=" << thread_id << ", part_length=" << part_length << ", layer_num=" << layer_num << "" << std::endl;
  
  for(int i = thread_id*part_length; i < (thread_id+1)*part_length; i++)
  {
    std::cout << "[LOG] parallel_comp: thread_id=" << thread_id << ", processing i=" << i << " (loop start)" << std::endl;
    
    double residual_cur = 0, residual_pre = 0;
    vector<IMUST> x_buf(WIN_SIZE);
    for(int j = 0; j < WIN_SIZE; j++)
    {
      x_buf[j].R = layer.pose_vec[i*GAP+j].q.toRotationMatrix();
      x_buf[j].p = layer.pose_vec[i*GAP+j].t;
    }
    
    size_t mem_cost = 0;
    for(int loop = 0; loop < layer.max_iter; loop++)
    {
      std::cout << "[LOG] parallel_comp: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << " started." << std::endl;
      
      // 每次迭代重新创建src_pc，避免累积内存
      vector<pcl::PointCloud<PointType>::Ptr> src_pc;
      src_pc.resize(WIN_SIZE);
      
      if(layer_num != 1)
      {
        std::cout << "[LOG] parallel_comp: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", using existing pcds (layer_num != 1)" << std::endl;
        for(int j = i*GAP; j < i*GAP+WIN_SIZE; j++)
          src_pc[j-i*GAP] = (*layer.pcds[j]).makeShared();
      }
      else
      {
        std::cout << "[LOG] parallel_comp: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", loading pcds from disk" << std::endl;
        // 每次迭代都重新加载点云，避免一次性加载大量点云到内存
        for(int j = i*GAP; j < i*GAP+WIN_SIZE; j++)
        {
          pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
          mypcl::loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
          src_pc[j-i*GAP] = pc;
        }
        std::cout << "[LOG] parallel_comp: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", pcds loaded successfully" << std::endl;
      }

      std::cout << "[LOG] parallel_comp: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", creating surf_map" << std::endl;
      unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;
      
      for(size_t j = 0; j < WIN_SIZE; j++)
      {
        if(layer.downsample_size > 0) 
        {
          // 直接在源点云上进行体素化，避免额外拷贝
          downsample_voxel(*src_pc[j], layer.downsample_size);
        }
        cut_voxel(surf_map, *src_pc[j], Eigen::Quaterniond(x_buf[j].R), x_buf[j].p,
                  j, layer.voxel_size, WIN_SIZE, layer.eigen_ratio);
      }
      std::cout << "[LOG] parallel_comp: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", cut_voxel completed, surf_map.size()=" << surf_map.size() << std::endl;
      
      // 清除不再需要的点云数据
      src_pc.clear();
      std::cout << "[LOG] parallel_comp: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", src_pc cleared" << std::endl;
      
      std::cout << "[LOG] parallel_comp: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", starting recut" << std::endl;
      for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        iter->second->recut();
      std::cout << "[LOG] parallel_comp: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", recut completed" << std::endl;
      
      std::cout << "[LOG] parallel_comp: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", starting tras_opt" << std::endl;
      VOX_HESS voxhess(WIN_SIZE);
      for(auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
        iter->second->tras_opt(voxhess);
      std::cout << "[LOG] parallel_comp: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", tras_opt completed" << std::endl;

      std::cout << "[LOG] parallel_comp: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", starting VOX_OPTIMIZER" << std::endl;
      VOX_OPTIMIZER opt_lsv(WIN_SIZE);
      opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
      PLV(6) hess_vec;
      opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);
      std::cout << "[LOG] parallel_comp: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", damping_iter completed, residual_cur=" << residual_cur << std::endl;

      // 及时释放八叉树内存
      std::cout << "[LOG] parallel_comp: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", releasing octree memory" << std::endl;
      for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        delete iter->second;
      surf_map.clear();
      std::cout << "[LOG] parallel_comp: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", octree memory released" << std::endl;
      
      if(loop > 0 && abs(residual_pre-residual_cur)/abs(residual_cur) < 0.05 || loop == layer.max_iter-1)
      {
        std::cout << "[LOG] parallel_comp: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", convergence achieved, breaking" << std::endl;
        
        if(layer.mem_costs[thread_id] < mem_cost) layer.mem_costs[thread_id] = mem_cost;
        
        // 只有在定义了FULL_HESS宏时才写入hessians向量
        #ifdef FULL_HESS
        std::cout << "[LOG] parallel_comp: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", writing hessians (FULL_HESS defined)" << std::endl;
        for(int j = 0; j < WIN_SIZE*(WIN_SIZE-1)/2; j++)
          layer.hessians[i*(WIN_SIZE-1)*WIN_SIZE/2+j] = hess_vec[j];
        #endif
        
        break;
      }
      residual_pre = residual_cur;
      std::cout << "[LOG] parallel_comp: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", iteration completed, residual_pre=" << residual_pre << std::endl;
    }
    
    // 只在优化完成后生成关键帧点云
    std::cout << "[LOG] parallel_comp: thread_id=" << thread_id << ", i=" << i << ", starting keyframe generation" << std::endl;
    pcl::PointCloud<PointType>::Ptr pc_keyframe(new pcl::PointCloud<PointType>);
    vector<pcl::PointCloud<PointType>::Ptr> src_pc;
    src_pc.resize(WIN_SIZE);
    
    if(layer_num != 1)
    {
      for(int j = i*GAP; j < i*GAP+WIN_SIZE; j++)
        src_pc[j-i*GAP] = (*layer.pcds[j]).makeShared();
    }
    else
    {
      // 重新加载点云用于生成关键帧
      for(int j = i*GAP; j < i*GAP+WIN_SIZE; j++)
      {
        pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
        mypcl::loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
        src_pc[j-i*GAP] = pc;
      }
    }
    
    for(size_t j = 0; j < WIN_SIZE; j++)
    {
      Eigen::Quaterniond q_tmp;
      Eigen::Vector3d t_tmp;
      assign_qt(q_tmp, t_tmp, Quaterniond(x_buf[0].R.inverse() * x_buf[j].R),
                x_buf[0].R.inverse() * (x_buf[j].p - x_buf[0].p));

      pcl::PointCloud<PointType>::Ptr pc_oneframe(new pcl::PointCloud<PointType>);
      mypcl::transform_pointcloud(*src_pc[j], *pc_oneframe, t_tmp, q_tmp);
      pc_keyframe = mypcl::append_cloud(pc_keyframe, *pc_oneframe);
    }
    
    // 清除不再需要的点云数据
    src_pc.clear();
    
    downsample_voxel(*pc_keyframe, 0.05);
    next_layer.pcds[i] = pc_keyframe;
    
    std::cout << "[LOG] parallel_comp: thread_id=" << thread_id << ", i=" << i << " (loop end) completed" << std::endl;
  }
  
  std::cout << "[LOG] parallel_comp: thread_id=" << thread_id << " finished." << std::endl;
}

void parallel_tail(LAYER& layer, int thread_id, LAYER& next_layer)
{
  std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << " started." << std::endl;
  
  int& part_length = layer.part_length;
  int& layer_num = layer.layer_num;
  int& left_gap_num = layer.left_gap_num;

  std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", part_length=" << part_length << ", layer_num=" << layer_num << ", left_gap_num=" << left_gap_num << std::endl;

  double load_t = 0, undis_t = 0, dsp_t = 0, cut_t = 0, recut_t = 0, total_t = 0,
    tran_t = 0, sol_t = 0, save_t = 0;
  
  if(layer.gap_num-(layer.thread_num-1)*part_length+1!=left_gap_num) {
    std::cout << "[ERROR] parallel_tail: gap_num calculation mismatch!" << std::endl;
    printf("THIS IS WRONG!\n");
  }

  for(uint i = thread_id*part_length; i < thread_id*part_length+left_gap_num; i++)
  {
    std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", processing i=" << i << " (main loop start)" << std::endl;
    printf("parallel computing %d\n", i);
    double t0, t1, t_begin;
    t_begin = get_current_time();

    double residual_cur = 0, residual_pre = 0;
    vector<IMUST> x_buf(WIN_SIZE);
    for(int j = 0; j < WIN_SIZE; j++)
    {
      x_buf[j].R = layer.pose_vec[i*GAP+j].q.toRotationMatrix();
      x_buf[j].p = layer.pose_vec[i*GAP+j].t;
    }
    
    vector<pcl::PointCloud<PointType>::Ptr> src_pc_initial;
    if(layer_num != 1)
    {
      t0 = get_current_time();
      src_pc_initial.resize(WIN_SIZE);
      for(int j = i*GAP; j < i*GAP+WIN_SIZE; j++)
        src_pc_initial[j-i*GAP] = (*layer.pcds[j]).makeShared();
      load_t += get_current_time()-t0;
    }

    size_t mem_cost = 0;
    for(int loop = 0; loop < layer.max_iter; loop++)
    {
      std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << " started." << std::endl;
      
      // 每次迭代重新创建src_pc，避免累积内存
      vector<pcl::PointCloud<PointType>::Ptr> src_pc;
      src_pc.resize(WIN_SIZE);
      
      if(layer_num != 1)
      {
        std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", using existing pcds (layer_num != 1)" << std::endl;
        for(int j = 0; j < WIN_SIZE; j++)
          src_pc[j] = (*src_pc_initial[j]).makeShared();
        std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", created src_pc from initial (layer_num != 1)" << std::endl;
      }
      else
      {
        // 每次迭代都重新加载点云，避免一次性加载大量点云到内存
        t0 = get_current_time();
        std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", loading pcds from disk" << std::endl;
        for(int j = i*GAP; j < i*GAP+WIN_SIZE; j++)
        {
          pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
          mypcl::loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
          src_pc[j-i*GAP] = pc;
        }
        load_t += get_current_time()-t0;
        std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", loaded pcds from disk" << std::endl;
      }

      std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", creating surf_map" << std::endl;
      unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;

      for(size_t j = 0; j < WIN_SIZE; j++)
      {
        t0 = get_current_time();
        if(layer.downsample_size > 0) 
        {
          // 直接在源点云上进行体素化，避免额外拷贝
          downsample_voxel(*src_pc[j], layer.downsample_size);
        }
        dsp_t += get_current_time()-t0;

        t0 = get_current_time();
        cut_voxel(surf_map, *src_pc[j], Quaterniond(x_buf[j].R), x_buf[j].p,
                  j, layer.voxel_size, WIN_SIZE, layer.eigen_ratio);
        cut_t += get_current_time()-t0;
      }
      std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", cut_voxel completed, surf_map.size()=" << surf_map.size() << std::endl;
      
      // 清除不再需要的点云数据
      src_pc.clear();
      std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", src_pc cleared" << std::endl;

      t0 = get_current_time();
      std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", starting recut" << std::endl;
      for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        iter->second->recut();
      recut_t += get_current_time()-t0;
      std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", recut completed" << std::endl;

      t0 = get_current_time();
      std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", starting tras_opt" << std::endl;
      VOX_HESS voxhess(WIN_SIZE);
      for(auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
        iter->second->tras_opt(voxhess);
      tran_t += get_current_time()-t0;
      std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", tras_opt completed" << std::endl;

      std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", starting VOX_OPTIMIZER" << std::endl;
      VOX_OPTIMIZER opt_lsv(WIN_SIZE);
      t0 = get_current_time();
      opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
      PLV(6) hess_vec;
      opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);
      sol_t += get_current_time()-t0;
      std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", damping_iter completed, residual_cur=" << residual_cur << std::endl;

      // 及时释放八叉树内存
      std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", releasing octree memory" << std::endl;
      for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        delete iter->second;
      surf_map.clear();
      std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", octree memory released" << std::endl;
            
      if(loop > 0 && abs(residual_pre-residual_cur)/abs(residual_cur) < 0.05 || loop == layer.max_iter-1)
      {
        std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", convergence achieved, breaking" << std::endl;
        
        if(layer.mem_costs[thread_id] < mem_cost) layer.mem_costs[thread_id] = mem_cost;

        // 只有在定义了FULL_HESS宏时才写入hessians向量
        #ifdef FULL_HESS
        if(i < thread_id*part_length+left_gap_num)
        {
          std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", writing hessians (FULL_HESS defined)" << std::endl;
          for(int j = 0; j < WIN_SIZE*(WIN_SIZE-1)/2; j++)
            layer.hessians[i*(WIN_SIZE-1)*WIN_SIZE/2+j] = hess_vec[j];
        }
        #endif

        break;
      }
      residual_pre = residual_cur;
      std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", i=" << i << ", loop=" << loop << ", iteration completed, residual_pre=" << residual_pre << std::endl;
    }
    
    // 只在优化完成后生成关键帧点云
    std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", i=" << i << ", starting keyframe generation" << std::endl;
    pcl::PointCloud<PointType>::Ptr pc_keyframe(new pcl::PointCloud<PointType>);
    vector<pcl::PointCloud<PointType>::Ptr> src_pc;
    src_pc.resize(WIN_SIZE);
    
    if(layer_num != 1)
    {
      for(int j = 0; j < WIN_SIZE; j++)
        src_pc[j] = (*src_pc_initial[j]).makeShared();
    }
    else
    {
      // 重新加载点云用于生成关键帧
      for(int j = i*GAP; j < i*GAP+WIN_SIZE; j++)
      {
        pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
        mypcl::loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
        src_pc[j-i*GAP] = pc;
      }
    }
    
    for(size_t j = 0; j < WIN_SIZE; j++)
    {
      t1 = get_current_time();
      Eigen::Quaterniond q_tmp;
      Eigen::Vector3d t_tmp;
      assign_qt(q_tmp, t_tmp, Quaterniond(x_buf[0].R.inverse() * x_buf[j].R),
                x_buf[0].R.inverse() * (x_buf[j].p - x_buf[0].p));

      pcl::PointCloud<PointType>::Ptr pc_oneframe(new pcl::PointCloud<PointType>);
      mypcl::transform_pointcloud(*src_pc[j], *pc_oneframe, t_tmp, q_tmp);
      pc_keyframe = mypcl::append_cloud(pc_keyframe, *pc_oneframe);
      save_t += get_current_time()-t1;
    }
    
    // 清除不再需要的点云数据
    src_pc.clear();
    src_pc_initial.clear();
    std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", i=" << i << ", cleared src_pc and src_pc_initial" << std::endl;
    
    t0 = get_current_time();
    downsample_voxel(*pc_keyframe, 0.05);
    dsp_t += get_current_time()-t0;

    t0 = get_current_time();
    next_layer.pcds[i] = pc_keyframe;
    save_t += get_current_time()-t0;
    
    total_t += get_current_time()-t_begin;
    
    std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", i=" << i << " (main loop end) completed" << std::endl;
  }
  if(layer.tail > 0)
  {
    std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", processing tail (layer.tail=" << layer.tail << ")" << std::endl;
    
    int i = thread_id*part_length+left_gap_num;
    std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", tail i=" << i << " started" << std::endl;

    double residual_cur = 0, residual_pre = 0;
    vector<IMUST> x_buf(layer.last_win_size);
    for(int j = 0; j < layer.last_win_size; j++)
    {
      x_buf[j].R = layer.pose_vec[i*GAP+j].q.toRotationMatrix();
      x_buf[j].p = layer.pose_vec[i*GAP+j].t;
    }

    vector<pcl::PointCloud<PointType>::Ptr> src_pc_initial;
    if(layer_num != 1)
    {
      src_pc_initial.resize(layer.last_win_size);
      for(int j = i*GAP; j < i*GAP+layer.last_win_size; j++)
        src_pc_initial[j-i*GAP] = (*layer.pcds[j]).makeShared();
      std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", tail i=" << i << ", loaded initial pcds (layer_num != 1)" << std::endl;
    }

    size_t mem_cost = 0;
    for(int loop = 0; loop < layer.max_iter; loop++)
    {
      std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", tail i=" << i << ", loop=" << loop << " started" << std::endl;
      
      // 每次迭代重新创建src_pc，避免累积内存
      vector<pcl::PointCloud<PointType>::Ptr> src_pc;
      src_pc.resize(layer.last_win_size);
      
      if(layer_num != 1)
      {
        for(int j = 0; j < layer.last_win_size; j++)
          src_pc[j] = (*src_pc_initial[j]).makeShared();
      }
      else
      {
        // 每次迭代都重新加载点云，避免一次性加载大量点云到内存
        std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", tail i=" << i << ", loop=" << loop << ", loading pcds from disk" << std::endl;
        for(int j = i*GAP; j < i*GAP+layer.last_win_size; j++)
        {
          pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
          mypcl::loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
          src_pc[j-i*GAP] = pc;
        }
        std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", tail i=" << i << ", loop=" << loop << ", loaded pcds from disk" << std::endl;
      }

      unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;

      for(size_t j = 0; j < layer.last_win_size; j++)
      {
        if(layer.downsample_size > 0) downsample_voxel(*src_pc[j], layer.downsample_size);
        cut_voxel(surf_map, *src_pc[j], Quaterniond(x_buf[j].R), x_buf[j].p,
                  j, layer.voxel_size, layer.last_win_size, layer.eigen_ratio);
      }
      std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", tail i=" << i << ", loop=" << loop << ", cut_voxel completed, surf_map.size()=" << surf_map.size() << std::endl;
      
      // 清除不再需要的点云数据
      src_pc.clear();
      std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", tail i=" << i << ", loop=" << loop << ", src_pc cleared" << std::endl;
      
      for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        iter->second->recut();
      
      VOX_HESS voxhess(layer.last_win_size);
      for(auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
        iter->second->tras_opt(voxhess);

      VOX_OPTIMIZER opt_lsv(layer.last_win_size);
      opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
      PLV(6) hess_vec;
      opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);

      // 及时释放八叉树内存
      for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        delete iter->second;
      surf_map.clear();
      
      if(loop > 0 && abs(residual_pre-residual_cur)/abs(residual_cur) < 0.05 || loop == layer.max_iter-1)
      {
        if(layer.mem_costs[thread_id] < mem_cost) layer.mem_costs[thread_id] = mem_cost;

        // 只有在定义了FULL_HESS宏时才写入hessians向量
        #ifdef FULL_HESS
        std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", tail i=" << i << ", loop=" << loop << ", writing hessians (FULL_HESS defined)" << std::endl;
        for(int j = 0; j < layer.last_win_size*(layer.last_win_size-1)/2; j++)
          layer.hessians[i*(WIN_SIZE-1)*WIN_SIZE/2+j] = hess_vec[j];
        #endif
        
        break;
      }
      residual_pre = residual_cur;
      std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", tail i=" << i << ", loop=" << loop << ", iteration completed, residual_pre=" << residual_pre << std::endl;
    }
    
    // 只在优化完成后生成关键帧点云
    std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", tail i=" << i << ", starting keyframe generation" << std::endl;
    pcl::PointCloud<PointType>::Ptr pc_keyframe(new pcl::PointCloud<PointType>);
    vector<pcl::PointCloud<PointType>::Ptr> src_pc;
    src_pc.resize(layer.last_win_size);
    
    if(layer_num != 1)
    {
      for(int j = 0; j < layer.last_win_size; j++)
        src_pc[j] = (*src_pc_initial[j]).makeShared();
    }
    else
    {
      // 重新加载点云用于生成关键帧
      for(int j = i*GAP; j < i*GAP+layer.last_win_size; j++)
      {
        pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
        mypcl::loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
        src_pc[j-i*GAP] = pc;
      }
    }
    
    for(size_t j = 0; j < layer.last_win_size; j++)
    {
      Eigen::Quaterniond q_tmp;
      Eigen::Vector3d t_tmp;
      assign_qt(q_tmp, t_tmp, Quaterniond(x_buf[0].R.inverse() * x_buf[j].R),
                x_buf[0].R.inverse() * (x_buf[j].p - x_buf[0].p));

      pcl::PointCloud<PointType>::Ptr pc_oneframe(new pcl::PointCloud<PointType>);
      mypcl::transform_pointcloud(*src_pc[j], *pc_oneframe, t_tmp, q_tmp);
      pc_keyframe = mypcl::append_cloud(pc_keyframe, *pc_oneframe);
    }
    
    // 清除不再需要的点云数据
    src_pc.clear();
    src_pc_initial.clear();
    
    downsample_voxel(*pc_keyframe, 0.05);
    next_layer.pcds[i] = pc_keyframe;
    
    std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << ", tail i=" << i << " completed" << std::endl;
  }
  
  printf("total time: %.2fs\n", total_t);
  printf("load pcd %.2fs %.2f%% | undistort pcd %.2fs %.2f%% | "
   "downsample %.2fs %.2f%% | cut voxel %.2fs %.2f%% | recut %.2fs %.2f%% | trans %.2fs %.2f%% | solve %.2fs %.2f%% | "
   "save pcd %.2fs %.2f%%\n",
    load_t, load_t/total_t*100, undis_t, undis_t/total_t*100,
    dsp_t, dsp_t/total_t*100, cut_t, cut_t/total_t*100, recut_t, recut_t/total_t*100, tran_t, tran_t/total_t*100, sol_t, sol_t/total_t*100,
    save_t, save_t/total_t*100);
    
  std::cout << "[LOG] parallel_tail: thread_id=" << thread_id << " finished." << std::endl;
}

double global_ba(LAYER& layer)
{
  std::cout << "[LOG] global_ba: started." << std::endl;
  
  int window_size = layer.pose_vec.size();
  std::cout << "[LOG] global_ba: window_size=" << window_size << std::endl;
  
  // 特殊情况处理：如果只有一个姿势或没有姿势，不需要优化
  if (window_size <= 1) {
    std::cout << "[LOG] global_ba: window_size=" << window_size << ", skipping optimization." << std::endl;
    std::cout << "[LOG] global_ba: Final residual: 0.0" << std::endl;
    std::cout << "[LOG] global_ba: finished." << std::endl;
    return 0.0;
  }
  
  // 保存初始位姿用于计算优化前后的变化
  vector<mypcl::pose> initial_poses(window_size);
  for(int i = 0; i < window_size; i++)
  {
    initial_poses[i] = layer.pose_vec[i];
  }
  
  vector<IMUST> x_buf(window_size);
  for(int i = 0; i < window_size; i++)
  {
    x_buf[i].R = layer.pose_vec[i].q.toRotationMatrix();
    x_buf[i].p = layer.pose_vec[i].t;
  }
  std::cout << "[LOG] global_ba: x_buf initialized." << std::endl;

  // 只保留初始点云拷贝，每次迭代重新创建src_pc
  vector<pcl::PointCloud<PointType>::Ptr> src_pc_initial;
  src_pc_initial.resize(window_size);
  
  // 检查layer.pcds是否为空，如果为空则自己加载点云
  if (layer.pcds.empty() || layer.pcds[0] == nullptr) {
    std::cout << "[LOG] global_ba: Loading point clouds for global BA..." << std::endl;
    for(int i = 0; i < window_size; i++)
    {
      pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
      mypcl::loadPCD(layer.data_path, pcd_name_fill_num, pc, i, "pcd/");
      src_pc_initial[i] = pc;
      if (i % 10 == 0) {
        std::cout << "[LOG] global_ba: Loaded point cloud " << i << " / " << window_size << std::endl;
      }
    }
    std::cout << "[LOG] global_ba: All point clouds loaded for global BA." << std::endl;
  } else {
    std::cout << "[LOG] global_ba: Using existing pcds from layer." << std::endl;
    for(int i = 0; i < window_size; i++)
      src_pc_initial[i] = (*layer.pcds[i]).makeShared();
    std::cout << "[LOG] global_ba: Copied existing pcds." << std::endl;
  }

  double residual_cur = 0, residual_pre = 0;
  size_t mem_cost = 0, max_mem = 0;
  double dsp_t = 0, cut_t = 0, recut_t = 0, tran_t = 0, sol_t = 0, t0;
  
  std::cout << "[LOG] global_ba: Starting optimization loop with max_iter=" << layer.max_iter << std::endl;
  
  for(int loop = 0; loop < layer.max_iter; loop++)
  {
    std::cout<<"---------------------"<<std::endl;
    std::cout<<"Iteration "<<loop<<std::endl;

    // 每次迭代重新创建src_pc，避免累积内存
    vector<pcl::PointCloud<PointType>::Ptr> src_pc;
    src_pc.resize(window_size);
    for(int i = 0; i < window_size; i++)
      src_pc[i] = (*src_pc_initial[i]).makeShared();
    
    std::cout << "[LOG] global_ba: Iteration " << loop << ", src_pc created." << std::endl;

    unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;
    std::cout << "[LOG] global_ba: Iteration " << loop << ", surf_map created." << std::endl;

    for(int i = 0; i < window_size; i++)
    {
      t0 = get_current_time();
      if(layer.downsample_size > 0) 
      {
        // 直接在源点云上进行体素化，避免额外拷贝
        downsample_voxel(*src_pc[i], layer.downsample_size);
      }
      dsp_t += get_current_time() - t0;
      t0 = get_current_time();
      cut_voxel(surf_map, *src_pc[i], Quaterniond(x_buf[i].R), x_buf[i].p, i,
                layer.voxel_size, window_size, layer.eigen_ratio*2);
      cut_t += get_current_time() - t0;
    }
    
    std::cout << "[LOG] global_ba: Iteration " << loop << ", cut_voxel completed, surf_map.size()=" << surf_map.size() << std::endl;
    
    // 清除不再需要的点云数据
    src_pc.clear();
    std::cout << "[LOG] global_ba: Iteration " << loop << ", src_pc cleared." << std::endl;
    
    t0 = get_current_time();
    std::cout << "[LOG] global_ba: Iteration " << loop << ", starting recut." << std::endl;
    for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      iter->second->recut();
    recut_t += get_current_time() - t0;
    std::cout << "[LOG] global_ba: Iteration " << loop << ", recut completed." << std::endl;
    
    t0 = get_current_time();
    std::cout << "[LOG] global_ba: Iteration " << loop << ", starting tras_opt." << std::endl;
    VOX_HESS voxhess(window_size);
    for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      iter->second->tras_opt(voxhess);
    tran_t += get_current_time() - t0;
    std::cout << "[LOG] global_ba: Iteration " << loop << ", tras_opt completed." << std::endl;
    
    t0 = get_current_time();
    std::cout << "[LOG] global_ba: Iteration " << loop << ", starting VOX_OPTIMIZER." << std::endl;
    VOX_OPTIMIZER opt_lsv(window_size);
    opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
    PLV(6) hess_vec;
    opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);
    sol_t += get_current_time() - t0;
    std::cout << "[LOG] global_ba: Iteration " << loop << ", damping_iter completed, residual_cur=" << residual_cur << std::endl;

    // 及时释放八叉树内存
    std::cout << "[LOG] global_ba: Iteration " << loop << ", releasing octree memory." << std::endl;
    for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      delete iter->second;
    surf_map.clear();
    std::cout << "[LOG] global_ba: Iteration " << loop << ", octree memory released." << std::endl;
    
    std::cout << "Residual absolute: " << abs(residual_pre-residual_cur) << " | "
      << "percentage: " << abs(residual_pre-residual_cur)/abs(residual_cur) << std::endl;
    
    if(loop > 0 && abs(residual_pre-residual_cur)/abs(residual_cur) < 0.05 || loop == layer.max_iter-1)
    {
      std::cout << "[LOG] global_ba: Iteration " << loop << ", convergence achieved, breaking." << std::endl;
      
      if(max_mem < mem_cost) max_mem = mem_cost;
      
      #ifdef FULL_HESS
      std::cout << "[LOG] global_ba: Iteration " << loop << ", writing hessians (FULL_HESS defined)." << std::endl;
      for(int i = 0; i < window_size*(window_size-1)/2; i++)
        layer.hessians[i] = hess_vec[i];
      #endif
      break;
    }
    residual_pre = residual_cur;
    std::cout << "[LOG] global_ba: Iteration " << loop << " completed, residual_pre=" << residual_pre << std::endl;
  }
  
  // 清除不再需要的点云数据
  src_pc_initial.clear();
  std::cout << "[LOG] global_ba: Cleared src_pc_initial." << std::endl;
  
  for(int i = 0; i < window_size; i++)
  {
    layer.pose_vec[i].q = Quaterniond(x_buf[i].R);
    layer.pose_vec[i].t = x_buf[i].p;
  }
  
  std::cout << "[LOG] global_ba: Updated layer pose_vec." << std::endl;
  printf("Downsample: %f, Cut: %f, Recut: %f, Tras: %f, Sol: %f\n", dsp_t, cut_t, recut_t, tran_t, sol_t);
  
  std::cout << "[LOG] global_ba: Final residual: " << residual_cur << std::endl;
  std::cout << "[LOG] global_ba: finished." << std::endl;
  
  // 计算优化前后的位姿变化
  double avg_translation_change = 0.0, max_translation_change = 0.0;
  double avg_rotation_change = 0.0, max_rotation_change = 0.0;
  
  for(int i = 0; i < window_size; i++)
  {
    // 计算平移变化
    Eigen::Vector3d trans_diff = layer.pose_vec[i].t - initial_poses[i].t;
    double trans_change = trans_diff.norm();
    avg_translation_change += trans_change;
    if(trans_change > max_translation_change) max_translation_change = trans_change;
    
    // 计算旋转变化（角度）
    Eigen::Quaterniond rot_diff = layer.pose_vec[i].q * initial_poses[i].q.inverse();
    double rot_angle = rot_diff.angularDistance(initial_poses[i].q) * 180.0 / M_PI;
    avg_rotation_change += rot_angle;
    if(rot_angle > max_rotation_change) max_rotation_change = rot_angle;
  }
  
  avg_translation_change /= window_size;
  avg_rotation_change /= window_size;
  
  std::cout << "[LOG] global_ba: Pose Change Statistics:" << std::endl;
  std::cout << "[LOG] global_ba:   Average translation change: " << avg_translation_change << " meters" << std::endl;
  std::cout << "[LOG] global_ba:   Maximum translation change: " << max_translation_change << " meters" << std::endl;
  std::cout << "[LOG] global_ba:   Average rotation change: " << avg_rotation_change << " degrees" << std::endl;
  std::cout << "[LOG] global_ba:   Maximum rotation change: " << max_rotation_change << " degrees" << std::endl;
  
  return residual_cur;
}

void distribute_thread(LAYER& layer, LAYER& next_layer)
{
  int& thread_num = layer.thread_num;
  std::cout << "[LOG] distribute_thread: Starting with thread_num=" << thread_num << std::endl;
  
  double t0 = get_current_time();
  for(int i = 0; i < thread_num; i++)
  {
    if(i < thread_num-1)
    {
      std::cout << "[LOG] distribute_thread: Creating thread " << i << " for parallel_comp." << std::endl;
      layer.mthreads[i] = new thread(parallel_comp, ref(layer), i, ref(next_layer));
    }
    else
    {
      std::cout << "[LOG] distribute_thread: Creating thread " << i << " for parallel_tail." << std::endl;
      layer.mthreads[i] = new thread(parallel_tail, ref(layer), i, ref(next_layer));
    }
  }
  std::cout << "[LOG] distribute_thread: All threads created. Time taken: " << get_current_time()-t0 << " seconds." << std::endl;

  t0 = get_current_time();
  for(int i = 0; i < thread_num; i++)
  {
    std::cout << "[LOG] distribute_thread: Joining thread " << i << "." << std::endl;
    layer.mthreads[i]->join();
    std::cout << "[LOG] distribute_thread: Thread " << i << " joined." << std::endl;
    delete layer.mthreads[i];
    std::cout << "[LOG] distribute_thread: Thread " << i << " deleted." << std::endl;
  }
  std::cout << "[LOG] distribute_thread: All threads joined and deleted. Time taken: " << get_current_time()-t0 << " seconds." << std::endl;
  std::cout << "[LOG] distribute_thread: Completed." << std::endl;
}

// 新增函数：使用pose.json解算所有帧点云
void full_cloud_solver(const std::string& data_path, int pcd_name_fill_num, bool use_timestamp = false, const std::string& custom_name = "")
{
  std::cout << "====================" << std::endl;
  std::cout << "Full Cloud Solver Mode" << std::endl;
  std::cout << "====================" << std::endl;
  std::cout << "Data Path: " << data_path << std::endl;
  std::cout << "PCD Name Fill Num: " << pcd_name_fill_num << std::endl;
  
  // 1. 读取pose.json文件
  std::vector<mypcl::pose> pose_vec = mypcl::read_pose(data_path + "pose.json");
  int frame_num = pose_vec.size();
  std::cout << "Loaded " << frame_num << " poses from pose.json" << std::endl;
  
  // 2. 创建输出点云
  pcl::PointCloud<PointType>::Ptr final_cloud(new pcl::PointCloud<PointType>);
  
  // 3. 逐帧加载点云并转换到全局坐标系
  std::cout << "Processing point clouds..." << std::endl;
  
  // 处理所有帧
  int process_frame_num = frame_num;
  
  for(int i = 0; i < process_frame_num; i++)
  {
    // 加载点云
    pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
    
    // 直接使用mypcl::loadPCD函数加载点云，与原始程序保持一致
    mypcl::loadPCD(data_path, pcd_name_fill_num, pc, i, "pcd/");
    
    if(pc->size() == 0) {
      // 如果加载失败或点云为空，跳过该帧
      continue;
    }
    
    // 直接使用位姿转换点云
    pcl::PointCloud<PointType> transformed_pc;
    transformed_pc.points.resize(pc->points.size());
    transformed_pc.width = pc->points.size();
    transformed_pc.height = 1;
    
    for(size_t j = 0; j < pc->points.size(); j++)
    {
      Eigen::Vector3d pt_cur(pc->points[j].x, pc->points[j].y, pc->points[j].z);
      Eigen::Vector3d pt_to = pose_vec[i].q * pt_cur + pose_vec[i].t;
      
      transformed_pc.points[j].x = pt_to.x();
      transformed_pc.points[j].y = pt_to.y();
      transformed_pc.points[j].z = pt_to.z();
    }
    
    // 合并到最终点云
    *final_cloud += transformed_pc;
    
    if(i % 10 == 0) {
      std::cout << "Processed frame " << i << " / " << process_frame_num << std::endl;
      std::cout << "Current final cloud size: " << final_cloud->size() << std::endl;
    }
  }
  
  // 4. 体素化下采样，减少点云数量
  std::cout << "Downsampling final point cloud..." << std::endl;
  
  // 确保点云不为空
  if(final_cloud->size() > 0) {
    // 创建下采样点云副本
    pcl::PointCloud<PointType>::Ptr downsampled_cloud(new pcl::PointCloud<PointType>);
    *downsampled_cloud = *final_cloud;
    
    // 对副本进行下采样
    downsample_voxel(*downsampled_cloud, 0.05); // 使用0.05米的体素大小
    
    // 5. 保存最终点云
    std::string output_file;
    
    if (use_timestamp) {
      std::string timestamp = mypcl::generate_timestamp();
      if (custom_name.empty()) {
        output_file = data_path + "full_cloud_" + timestamp + ".pcd";
      } else {
        output_file = data_path + custom_name + "_" + timestamp + ".pcd";
      }
    } else {
      if (custom_name.empty()) {
        output_file = data_path + "full_cloud.pcd";
      } else {
        output_file = data_path + custom_name + ".pcd";
      }
    }
    
    std::cout << "Saving final point cloud to " << output_file << std::endl;
    std::cout << "Original points: " << final_cloud->size() << std::endl;
    std::cout << "Downsampled points: " << downsampled_cloud->size() << std::endl;
    
    // 保存下采样后的点云
    pcl::io::savePCDFileBinary(output_file, *downsampled_cloud);
    
    std::cout << "====================" << std::endl;
    std::cout << "Full Cloud Solver Complete!" << std::endl;
    std::cout << "Original points: " << final_cloud->size() << std::endl;
    std::cout << "Output file: " << output_file << std::endl;
    std::cout << "====================" << std::endl;
  } else {
    std::cout << "====================" << std::endl;
    std::cout << "Full Cloud Solver Complete!" << std::endl;
    std::cout << "No points were processed!" << std::endl;
    std::cout << "====================" << std::endl;
  }
}

int main(int argc, char** argv)
{
  std::cout << "[LOG] Program started. argv[0]: " << argv[0] << std::endl;
  
  // 新增命令行参数：支持full_cloud_solver模式和增强的HBA模式
  if (argc < 2) {
    std::cerr << "Usage:" << std::endl;
    std::cerr << "  1. Enhanced HBA mode: " << argv[0] << " <total_layer_num> <pcd_name_fill_num> <data_path> <thread_num> [options]" << std::endl;
    std::cerr << "     Options:" << std::endl;
    std::cerr << "       --timestamp          Output files with timestamp markers" << std::endl;
    std::cerr << "       --custom-name <name> Use custom name prefix for output files" << std::endl;
    std::cerr << "       --full-cloud         Generate full point cloud with timestamp markers" << std::endl;
    std::cerr << "  2. Full cloud solver mode: " << argv[0] << " full <pcd_name_fill_num> <data_path> [options]" << std::endl;
    std::cerr << "     Options:" << std::endl;
    std::cerr << "       --timestamp          Output files with timestamp markers" << std::endl;
    std::cerr << "       --custom-name <name> Use custom name prefix for output files" << std::endl;
    return 1;
  }

  std::string mode = argv[1];
  std::cout << "[LOG] Mode: " << mode << std::endl;
  
  // 检查是否为full_cloud_solver模式
  if (mode == "full") {
    if (argc < 4) {
      std::cerr << "Usage for full cloud solver: " << argv[0] << " full <pcd_name_fill_num> <data_path> [--timestamp] [--custom-name <name>]" << std::endl;
      return 1;
    }
    
    int pcd_name_fill_num = std::stoi(argv[2]);
    std::string data_path = argv[3];
    std::cout << "[LOG] full_cloud_solver mode parameters: pcd_name_fill_num=" << pcd_name_fill_num << ", data_path=" << data_path << std::endl;
    
    // 解析可选参数
    bool use_timestamp = false;
    std::string custom_name = "";
    
    for (int i = 4; i < argc; i++) {
      std::string arg = argv[i];
      if (arg == "--timestamp") {
        use_timestamp = true;
        std::cout << "[LOG] Option: --timestamp" << std::endl;
      } else if (arg == "--custom-name" && i + 1 < argc) {
        custom_name = argv[++i];
        std::cout << "[LOG] Option: --custom-name=" << custom_name << std::endl;
      }
    }
    
    // 调用full_cloud_solver函数
    std::cout << "[LOG] Calling full_cloud_solver function..." << std::endl;
    full_cloud_solver(data_path, pcd_name_fill_num, use_timestamp, custom_name);
    std::cout << "[LOG] full_cloud_solver function returned." << std::endl;
    return 0;
  }
  
  // 原始HBA模式需要至少5个参数
  if (argc < 5) {
    std::cerr << "Usage for enhanced HBA mode: " << argv[0] << " <total_layer_num> <pcd_name_fill_num> <data_path> <thread_num> [--timestamp] [--custom-name <name>] [--full-cloud]" << std::endl;
    return 1;
  }
  
  // 解析必需参数
  int total_layer_num = std::stoi(argv[1]);
  pcd_name_fill_num = std::stoi(argv[2]);
  std::string data_path = argv[3];
  int thread_num = std::stoi(argv[4]);
  
  std::cout << "[LOG] Enhanced HBA mode parameters: total_layer_num=" << total_layer_num << ", pcd_name_fill_num=" << pcd_name_fill_num << ", data_path=" << data_path << ", thread_num=" << thread_num << std::endl;
  
  // 解析可选参数
  bool use_timestamp = false;
  std::string custom_name = "";
  bool generate_full_cloud = false;
  
  for (int i = 5; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--timestamp") {
      use_timestamp = true;
      std::cout << "[LOG] Option: --timestamp" << std::endl;
    } else if (arg == "--custom-name" && i + 1 < argc) {
      custom_name = argv[++i];
      std::cout << "[LOG] Option: --custom-name=" << custom_name << std::endl;
    } else if (arg == "--full-cloud") {
      generate_full_cloud = true;
      std::cout << "[LOG] Option: --full-cloud" << std::endl;
    }
  }

  std::cout << "HBA Parameters:" << std::endl;
  std::cout << "- total_layer_num: " << total_layer_num << std::endl;
  std::cout << "- pcd_name_fill_num: " << pcd_name_fill_num << std::endl;
  std::cout << "- data_path: " << data_path << std::endl;
  std::cout << "- thread_num: " << thread_num << std::endl;
  std::cout << "- use_timestamp: " << (use_timestamp ? "true" : "false") << std::endl;
  std::cout << "- custom_name: " << custom_name << std::endl;
  std::cout << "- generate_full_cloud: " << (generate_full_cloud ? "true" : "false") << std::endl;

  std::cout << "[LOG] Creating HBA object..." << std::endl;
  HBA hba(total_layer_num, data_path, thread_num);
  std::cout << "[LOG] HBA object created." << std::endl;
  
  for(int i = 0; i < total_layer_num-1; i++)
  {
    std::cout << "[LOG] Layer " << i << " processing started." << std::endl;
    std::cout<<"---------------------"<<std::endl;
    
    // 确保下一层的pose_vec已经正确初始化
    if (hba.layers[i+1].pose_vec.empty()) {
      std::cout << "[LOG] Initializing next layer pose_vec..." << std::endl;
      // 计算下一层需要的pose数量
      int next_layer_pose_size = 0;
      if (i == total_layer_num - 2) {
        // 最后一层，只需要一个姿势
        next_layer_pose_size = 1;
      } else {
        // 其他层，需要计算关键帧数量
        next_layer_pose_size = (hba.layers[i].thread_num - 1) * hba.layers[i].part_length + hba.layers[i].left_gap_num;
        if (hba.layers[i].tail > 0) {
          next_layer_pose_size += 1;
        }
      }
      std::cout << "[LOG] Next layer pose_vec size: " << next_layer_pose_size << std::endl;
      hba.layers[i+1].pose_vec.resize(next_layer_pose_size);
    }
    
    distribute_thread(hba.layers[i], hba.layers[i+1]);
    std::cout << "[LOG] distribute_thread completed for layer " << i << "." << std::endl;
    
    std::cout << "[LOG] Calling update_next_layer_state for layer " << i << "..." << std::endl;
    hba.update_next_layer_state(i);
    std::cout << "[LOG] update_next_layer_state completed for layer " << i << "." << std::endl;
  }
  
  // 调用global_ba并获取最终残差
  std::cout << "[LOG] Calling global_ba..." << std::endl;
  double final_global_residual = global_ba(hba.layers[total_layer_num-1]);
  std::cout << "[LOG] global_ba completed. Final residual: " << final_global_residual << std::endl;
  
  // 调用增强版pose_graph_optimization，支持时间戳和完整点云生成
  std::cout << "[LOG] Calling pose_graph_optimization..." << std::endl;
  
  // 直接调用write_pose生成轨迹文件，跳过pose_graph_optimization以避免内存崩溃
  std::cout << "[LOG] Skipping pose_graph_optimization, writing pose directly..." << std::endl;
  mypcl::write_pose(hba.layers[0].pose_vec, hba.data_path, use_timestamp, custom_name);
  
  // 跳过完整点云生成，避免内存崩溃
  if (generate_full_cloud) {
    std::cout << "[LOG] Skipping full cloud generation to avoid memory crashes..." << std::endl;
    std::cout << "[LOG] Pose file has been generated successfully." << std::endl;
  }
  
  std::cout << "[LOG] pose_graph_optimization completed (skipped)." << std::endl;
  
  // 计算轨迹的整体精度指标
  const auto& final_layer = hba.layers[total_layer_num-1];
  size_t total_frames = final_layer.pose_vec.size();
  
  // 计算相邻帧之间的平移和旋转变化
  double avg_translation_step = 0.0;
  double avg_rotation_step = 0.0;
  double max_translation_step = 0.0;
  double max_rotation_step = 0.0;
  
  if (total_frames > 1) {
    for (size_t i = 0; i < total_frames - 1; i++) {
      // 计算相邻帧之间的平移变化
      Eigen::Vector3d trans_diff = final_layer.pose_vec[i+1].t - final_layer.pose_vec[i].t;
      double trans_step = trans_diff.norm();
      avg_translation_step += trans_step;
      if (trans_step > max_translation_step) max_translation_step = trans_step;
      
      // 计算相邻帧之间的旋转变化（角度）
      Eigen::Quaterniond rot_diff = final_layer.pose_vec[i+1].q * final_layer.pose_vec[i].q.inverse();
      double rot_step = rot_diff.angularDistance(final_layer.pose_vec[i].q) * 180.0 / M_PI;
      avg_rotation_step += rot_step;
      if (rot_step > max_rotation_step) max_rotation_step = rot_step;
    }
    
    avg_translation_step /= (total_frames - 1);
    avg_rotation_step /= (total_frames - 1);
  }
  
  // 计算轨迹的总长度
  double total_trajectory_length = 0.0;
  if (total_frames > 1) {
    for (size_t i = 0; i < total_frames - 1; i++) {
      Eigen::Vector3d trans_diff = final_layer.pose_vec[i+1].t - final_layer.pose_vec[i].t;
      total_trajectory_length += trans_diff.norm();
    }
  }
  
  // 输出详细的精度相关信息
  std::cout << "====================" << std::endl;
  std::cout << "HBA Trajectory Optimization Results" << std::endl;
  std::cout << "====================" << std::endl;
  std::cout << "Configuration Parameters:" << std::endl;
  std::cout << "  Data Path: " << data_path << std::endl;
  std::cout << "  Total Layer Num: " << total_layer_num << std::endl;
  std::cout << "  Thread Num: " << thread_num << std::endl;
  std::cout << "  PCD Name Fill Num: " << pcd_name_fill_num << std::endl;
  std::cout << "====================" << std::endl;
  std::cout << "Optimization Precision Metrics:" << std::endl;
  std::cout << "  Final Global BA Residual: " << std::fixed << std::setprecision(6) << final_global_residual << std::endl;
  std::cout << "  Total Optimized Frames: " << total_frames << std::endl;
  std::cout << "  Total Trajectory Length: " << std::fixed << std::setprecision(3) << total_trajectory_length << " meters" << std::endl;
  std::cout << "====================" << std::endl;
  std::cout << "Frame-to-Frame Consistency:" << std::endl;
  std::cout << "  Average Translation Step: " << std::fixed << std::setprecision(4) << avg_translation_step << " meters" << std::endl;
  std::cout << "  Maximum Translation Step: " << std::fixed << std::setprecision(4) << max_translation_step << " meters" << std::endl;
  std::cout << "  Average Rotation Step: " << std::fixed << std::setprecision(4) << avg_rotation_step << " degrees" << std::endl;
  std::cout << "  Maximum Rotation Step: " << std::fixed << std::setprecision(4) << max_rotation_step << " degrees" << std::endl;
  std::cout << "====================" << std::endl;
  std::cout << "Output Information:" << std::endl;
  std::cout << "  Trajectory optimization completed successfully!" << std::endl;
  std::cout << "  Optimized poses saved to: " << data_path << "pose.json" << std::endl;
  
  if (generate_full_cloud) {
    std::cout << "  Full point cloud generated!" << std::endl;
    std::cout << "  Check the output file in the data directory." << std::endl;
  }
  
  std::cout << "====================" << std::endl;
  
  printf("iteration complete\n");
  std::cout << "[LOG] Program completed successfully." << std::endl;
  return 0;
}