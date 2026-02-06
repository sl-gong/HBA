# HBA 算法中位姿图优化代码解读

## 代码概述

本文档解读 `/Users/gsl/work/slam/HBA/include/hba.hpp` 文件中第 196-283 行的 `pose_graph_optimization` 函数代码。该函数是 HBA（Hierarchical Bundle Adjustment）算法的核心组成部分，负责使用 GTSAM 库进行位姿图优化，提高轨迹估计的精度。

## 函数实现分析

### 1. 数据准备

```cpp
std::vector<mypcl::pose> upper_pose, init_pose;
upper_pose = layers[total_layer_num-1].pose_vec;
init_pose = layers[0].pose_vec;
std::vector<VEC(6)> upper_cov, init_cov;
upper_cov = layers[total_layer_num-1].hessians;
init_cov = layers[0].hessians;
```

**代码分析**：
- `upper_pose`：获取最高层（最稀疏）的优化后轨迹
- `init_pose`：获取底层（最密集）的原始轨迹
- `upper_cov`：获取最高层的 Hessian 矩阵（用于计算协方差）
- `init_cov`：获取底层的 Hessian 矩阵

**技术要点**：
- HBA 算法采用分层结构，每层处理不同密度的轨迹
- 最高层轨迹最稀疏但精度最高，底层轨迹最密集但精度较低
- 使用 Hessian 矩阵来估计轨迹的不确定性（协方差）

### 2. GTSAM 初始化

```cpp
int cnt = 0;
gtsam::Values initial;
gtsam::NonlinearFactorGraph graph;
gtsam::Vector Vector6(6);
Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8;
gtsam::noiseModel::Diagonal::shared_ptr priorModel = gtsam::noiseModel::Diagonal::Variances(Vector6);
initial.insert(0, gtsam::Pose3(gtsam::Rot3(init_pose[0].q.toRotationMatrix()), gtsam::Point3(init_pose[0].t)));
graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, gtsam::Pose3(gtsam::Rot3(init_pose[0].q.toRotationMatrix()),
                                                           gtsam::Point3(init_pose[0].t)), priorModel));
```

**代码分析**：
- `initial`：存储初始位姿估计
- `graph`：构建因子图
- `Vector6`：定义先验噪声模型的方差（位置：1e-6，旋转：1e-8）
- 添加第一个位姿的先验因子，固定轨迹的全局位置和朝向

**技术要点**：
- 使用 GTSAM 库进行非线性最小二乘优化
- 先验因子确保轨迹有全局参考系，避免漂移
- 噪声模型参数影响优化的稳定性和精度

### 3. 添加底层轨迹的因子

```cpp
for(uint i = 0; i < init_pose.size(); i++)
{
  if(i > 0) initial.insert(i, gtsam::Pose3(gtsam::Rot3(init_pose[i].q.toRotationMatrix()), gtsam::Point3(init_pose[i].t)));

  if(i%GAP == 0 && cnt < init_cov.size())
    for(int j = 0; j < WIN_SIZE-1; j++)
      for(int k = j+1; k < WIN_SIZE; k++)
      {
        if(i+j+1 >= init_pose.size() || i+k >= init_pose.size()) break;

        cnt++;
        if(init_cov[cnt-1].norm() < 1e-20) continue;

        Eigen::Vector3d t_ab = init_pose[i+j].t;
        Eigen::Matrix3d R_ab = init_pose[i+j].q.toRotationMatrix();
        t_ab = R_ab.transpose() * (init_pose[i+k].t - t_ab);
        R_ab = R_ab.transpose() * init_pose[i+k].q.toRotationMatrix();
        gtsam::Rot3 R_sam(R_ab);
        gtsam::Point3 t_sam(t_ab);
        
        Vector6 << fabs(1.0/init_cov[cnt-1](0)), fabs(1.0/init_cov[cnt-1](1)), fabs(1.0/init_cov[cnt-1](2)),
                   fabs(1.0/init_cov[cnt-1](3)), fabs(1.0/init_cov[cnt-1](4)), fabs(1.0/init_cov[cnt-1](5));
        gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);
        gtsam::NonlinearFactor::shared_ptr factor(new gtsam::BetweenFactor<gtsam::Pose3>(i+j, i+k, gtsam::Pose3(R_sam, t_sam),
                                                  odometryNoise));
        graph.push_back(factor);
      }
}
```

**代码分析**：
- 遍历底层轨迹的每个位姿，每隔 `GAP` 个点处理一次
- 对每个处理点，在窗口大小 `WIN_SIZE` 内添加相邻位姿之间的相对约束
- 计算相对位姿（位置和旋转），并根据 Hessian 矩阵计算噪声协方差
- 创建 `BetweenFactor` 并添加到因子图中

**技术要点**：
- 使用滑动窗口方法处理底层轨迹的相对约束
- 通过 Hessian 矩阵的逆来估计相对位姿的不确定性
- 跳过 Hessian 矩阵范数过小的约束，避免数值不稳定

### 4. 添加最高层轨迹的因子

```cpp
int pose_size = upper_pose.size();
cnt = 0;
for(int i = 0; i < pose_size-1; i++)
  for(int j = i+1; j < pose_size; j++)
  {
    cnt++;
    if(upper_cov[cnt-1].norm() < 1e-20) continue;

    Eigen::Vector3d t_ab = upper_pose[i].t;
    Eigen::Matrix3d R_ab = upper_pose[i].q.toRotationMatrix();
    t_ab = R_ab.transpose() * (upper_pose[j].t - t_ab);
    R_ab = R_ab.transpose() * upper_pose[j].q.toRotationMatrix();
    gtsam::Rot3 R_sam(R_ab);
    gtsam::Point3 t_sam(t_ab);

    Vector6 << fabs(1.0/upper_cov[cnt-1](0)), fabs(1.0/upper_cov[cnt-1](1)), fabs(1.0/upper_cov[cnt-1](2)),
               fabs(1.0/upper_cov[cnt-1](3)), fabs(1.0/upper_cov[cnt-1](4)), fabs(1.0/upper_cov[cnt-1](5));
    gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);
    gtsam::NonlinearFactor::shared_ptr factor(new gtsam::BetweenFactor<gtsam::Pose3>(i*pow(GAP, total_layer_num-1),
                                              j*pow(GAP, total_layer_num-1), gtsam::Pose3(R_sam, t_sam), odometryNoise));
    graph.push_back(factor);
  }
```

**代码分析**：
- 遍历最高层轨迹的所有位姿对
- 计算每对位姿之间的相对约束
- 根据最高层的 Hessian 矩阵计算噪声协方差
- 创建 `BetweenFactor` 并添加到因子图中，注意这里的位姿索引需要根据层级进行缩放

**技术要点**：
- 最高层轨迹更稀疏但精度更高，因此其约束权重更大
- 通过 `pow(GAP, total_layer_num-1)` 计算位姿在底层轨迹中的实际索引
- 同样跳过 Hessian 矩阵范数过小的约束

### 5. 执行位姿图优化

```cpp
gtsam::ISAM2Params parameters;
parameters.relinearizeThreshold = 0.01;
parameters.relinearizeSkip = 1;
gtsam::ISAM2 isam(parameters);
isam.update(graph, initial);
isam.update();

gtsam::Values results = isam.calculateEstimate();
```

**代码分析**：
- 配置 ISAM2（Incremental Smoothing and Mapping）参数
- `relinearizeThreshold`：设置重线性化阈值，控制优化精度和速度的平衡
- `relinearizeSkip`：设置重线性化间隔
- 执行两次 `update` 操作，确保优化收敛
- 获取优化后的位姿估计结果

**技术要点**：
- ISAM2 是 GTSAM 库中用于实时位姿图优化的高效算法
- 重线性化参数的设置对优化性能有重要影响
- 两次 `update` 操作可以提高优化的收敛性

### 6. 更新轨迹并保存结果

```cpp
cout << "vertex size " << results.size() << endl;

for(uint i = 0; i < results.size(); i++)
{
  gtsam::Pose3 pose = results.at(i).cast<gtsam::Pose3>();
  assign_qt(init_pose[i].q, init_pose[i].t, Eigen::Quaterniond(pose.rotation().matrix()), pose.translation());
}
mypcl::write_pose(init_pose, data_path);
printf("pgo complete\n");
```

**代码分析**：
- 输出优化后的顶点数量
- 将 GTSAM 优化结果转换回 HBA 内部的位姿格式
- 更新底层轨迹的位姿
- 将优化后的轨迹保存到文件

**技术要点**：
- 使用 `assign_qt` 函数确保四元数和位置的正确转换
- 保存优化后的轨迹，供后续点云生成使用

## 技术要点总结

### 1. 分层优化策略

HBA 算法采用分层结构，通过 `pose_graph_optimization` 函数实现不同层级轨迹的融合：
- **底层轨迹**：密度高，包含所有原始位姿，但精度较低
- **最高层轨迹**：密度低，但经过多层优化后精度较高
- **融合策略**：将最高层的高精度约束应用到底层轨迹上，提高整体精度

### 2. 噪声模型估计

代码使用 Hessian 矩阵来估计位姿的不确定性：
- Hessian 矩阵的对角线元素表示各自由度的二阶导数
- 通过 `1.0/init_cov[cnt-1](i)` 计算噪声协方差
- 绝对值操作确保协方差为正

### 3. 因子图构建

因子图是位姿图优化的核心数据结构：
- **PriorFactor**：添加先验约束，固定全局参考系
- **BetweenFactor**：添加相邻位姿之间的相对约束
- **约束选择**：通过 `GAP` 间隔和 `WIN_SIZE` 窗口控制约束密度

### 4. ISAM2 优化参数

ISAM2 优化参数的设置对性能影响显著：
- `relinearizeThreshold = 0.01`：平衡优化精度和计算速度
- `relinearizeSkip = 1`：每次迭代都进行重线性化，提高精度

## 代码优化建议

### 1. 数值稳定性改进

```cpp
// 原代码
if(init_cov[cnt-1].norm() < 1e-20) continue;

// 优化建议
const double COV_THRESHOLD = 1e-20;
if(init_cov[cnt-1].norm() < COV_THRESHOLD || 
   std::isnan(init_cov[cnt-1](0)) || 
   std::isinf(init_cov[cnt-1](0))) continue;
```

**优化理由**：添加对 NaN 和 Inf 值的检查，提高数值稳定性。

### 2. 计算效率优化

```cpp
// 原代码
for(int i = 0; i < pose_size-1; i++)
  for(int j = i+1; j < pose_size; j++)
  {
    // 计算所有位姿对之间的约束
  }

// 优化建议
const int MAX_EDGE_DISTANCE = 5; // 最大边距离
for(int i = 0; i < pose_size-1; i++)
  for(int j = i+1; j < std::min(i+MAX_EDGE_DISTANCE+1, pose_size); j++)
  {
    // 只计算邻近位姿之间的约束
  }
```

**优化理由**：限制最大边距离，减少因子图规模，提高优化速度，同时保持优化精度。

### 3. 参数配置灵活性

```cpp
// 原代码
gtsam::ISAM2Params parameters;
parameters.relinearizeThreshold = 0.01;
parameters.relinearizeSkip = 1;

// 优化建议
struct ISAM2Config {
  double relinearizeThreshold = 0.01;
  int relinearizeSkip = 1;
  // 其他参数...
};

ISAM2Config config;
gtsam::ISAM2Params parameters;
parameters.relinearizeThreshold = config.relinearizeThreshold;
parameters.relinearizeSkip = config.relinearizeSkip;
```

**优化理由**：将 ISAM2 参数封装到配置结构中，提高代码的可维护性和参数调优的灵活性。

## 输入输出示例

### 输入输出示例

**输入**：
- 分层优化后的轨迹数据：
  - 底层轨迹：3407 个位姿
  - 最高层轨迹：681 个位姿
- 对应的 Hessian 矩阵数据

**输出**：
- 优化后的轨迹文件：`pose.json`
- 优化过程日志：

```
vertex size 3407
pgo complete
```

**优化效果**：
- 位置误差：接近 0
- 旋转误差：平均约 0.346 度
- 点云精度：使用优化后的轨迹生成的点云更加准确对齐

## 总结

`pose_graph_optimization` 函数是 HBA 算法中的关键组件，通过以下步骤实现高精度轨迹估计：

1. **数据准备**：获取不同层级的轨迹和协方差数据
2. **因子图构建**：添加先验约束和相对约束
3. **优化执行**：使用 ISAM2 算法进行位姿图优化
4. **结果更新**：将优化结果应用到底层轨迹

该实现充分利用了 GTSAM 库的强大功能，结合 HBA 算法的分层优化策略，实现了高精度的轨迹估计，为后续的点云生成提供了可靠的位姿信息。

通过本文档的解读，希望能帮助读者理解 HBA 算法中位姿图优化的实现原理，为后续的算法改进和应用提供参考。