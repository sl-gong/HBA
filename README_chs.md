# HBA: 一种全局一致且高效的大规模激光雷达建图模块

## **1. 简介**
**HBA** 旨在解决位姿图优化（PGO）后点云地图内的不一致性无法完全消除的问题。尽管 PGO 具有较高的时间效率，但它并不能直接优化建图的一致性。激光雷达束调整（BA）可以缓解这一问题；然而，在大规模地图上，它的计算时间过长。HBA 提出了一种分层结构，将庞大的激光雷达 BA 问题分解为多个较小的 BA 问题，并使用 PGO 平滑更新整个激光雷达位姿。与原始 BA 方法相比，HBA 可以实现相似的精度，但计算时间要少得多。

<div align="center">
  <div align="center">
    <img src="figure/lidar_frame.png"  width="100%" />
  </div>
  <font color=#a0a0a0 size=2>我们提出的分层束调整的金字塔结构。</font>
</div>

## **2. 论文和视频**
我们的论文已发表在 [IEEE RA-L](https://ieeexplore.ieee.org/abstract/document/10024300) 上，相应的视频可以在 [Bilibili](https://www.bilibili.com/video/BV1Qg41127j9/?spm_id_from=333.999.0.0) 或 [YouTube](https://youtu.be/CuLnTnXVujw) 上观看。如果您发现我们的工作对您的研究有用，请考虑引用：

```
@ARTICLE{10024300,
  author={Liu, Xiyuan and Liu, Zheng and Kong, Fanze and Zhang, Fu},
  journal={IEEE Robotics and Automation Letters}, 
  title={Large-Scale LiDAR Consistent Mapping Using Hierarchical LiDAR Bundle Adjustment}, 
  year={2023},
  volume={8},
  number={3},
  pages={1523-1530},
  doi={10.1109/LRA.2023.3238902}}
```

## **3. 运行代码**
### 3.1 前置条件
我们的代码已在以下环境中测试通过：
- [Ubuntu 20.04](https://releases.ubuntu.com/focal/) 搭配 [ROS Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu)
- [Ubuntu 18.04](https://releases.ubuntu.com/18.04/) 搭配 [ROS Melodic](https://wiki.ros.org/melodic/Installation/Ubuntu)
- [PCL 1.10.0](https://github.com/PointCloudLibrary/pcl/releases)
- [Eigen 3.3.7](https://gitlab.com/libeigen/eigen/-/releases/3.3.7)
- [GTSAM 4.1.1](https://github.com/borglab/gtsam)

### 3.2 文件结构
要在您自己的数据上测试，请按照以下结构准备文件：一个名为 `pcd` 的文件夹，包含点云文件；一个 `pose.json` 文件，包含每个激光雷达扫描的初始位姿。注意位姿的格式为 `tx ty tz qw qx qy qz`（平移和四元数）。

```
.
├── pcd
│   ├── 0.pcd
│   └── 1.pcd
└── pose.json
```

### 3.3 重要参数
#### 在 `hba.launch` 文件中
* `total_layer_num`：HBA 中使用的层数。默认值为 `3`。
* `pcd_name_fill_num`：pcd 文件名中的前缀零个数（例如，如果 pcd 文件以 `00000.pcd` 开头，请填写 `5`）。默认值为 `0`。
* `thread_num`：CPU 并行计算中使用的线程数。默认值为 `16`。

#### 在 `hba.hpp` 文件中
* `downsample_size`：每个激光雷达扫描的点云下采样叶大小。默认值为 `0.1`。
* `voxel_size`：激光雷达 BA 中使用的初始体素大小。默认值为 `4.0`。
* `eigen_ratio`：用于判断该体素是否包含有效平面特征的阈值。值越大，阈值越宽松。默认值为 `0.1`。
* `reject_ratio`：用于在优化中拒绝最大残留误差的体素比例的阈值。默认值为 `0.05`。

#### 在 `ba.hpp` 文件中
* `WIN_SIZE`：局部 BA 中使用的窗口大小。默认值为 `10`。
* `GAP`：两个相邻窗口起始位置之间的步长。默认值为 `5`。
* `layer_limit`：激光雷达 BA 中体素重新分割的最大次数。默认值为 `2`。

备注：在 `global_ba` 函数中，我们使用了比局部 BA 稍大的 `eigen_ratio=0.2` 参数，这通常会导致更快的收敛。您可以在 `hba.cpp` 文件的 `cut_voxel` 函数中调整这些参数。较小的体素大小和特征值比例参数通常会导致更高的精度，但计算时间更长。

### 3.4 尝试我们的数据
我们自行采集的 [park](https://drive.google.com/file/d/1vjmTiNULlrZ_7FMSDDy7w2Xw0_B1oz8D/view?usp=sharing) 数据集和公开的 [KITTI 07](https://drive.google.com/file/d/16Cck3c6ie_GT5HXHTy5VlxJc5vPC1O4r/view?usp=sharing) 数据集的压缩 pcd 文件及其初始位姿已上传到 OneDrive。您可以下载它们并使用提供的参数直接运行代码。

备注：`hba.launch` 仅优化激光雷达位姿，不可视化点云地图；`visualize.launch` 用于查看点云地图。另外，当您启动 `hba.launch` 时，启动完成后它只优化一次位姿。如果您对结果不满意，可以再次执行启动命令。

## **4. 应用**

### 4.1 全局优化点云建图一致性

#### 4.1.1 里程计闭环（请参见下方 KITTI 序列的结果）

<div align="center"><img src="figure/kitti.gif"  width="100%" /></div>
<div align="center"><img src="figure/kitti05.jpg"  width="100%" /></div>

#### 4.1.2 进一步优化建图一致性（请参见我们自行采集数据集的结果）

<div align="center"><img src="figure/lg02.jpg"  width="100%" /></div>

### 4.2 提供厘米级精度的点云地图

[MARSIM](https://github.com/hku-mars/MARSIM) 是一个用于基于激光雷达的无人机的轻量级点云真实模拟器，**HBA** 已为其贡献了十多个具有厘米级精度的真实世界点云地图。

## **5. 致谢**
在 **HBA** 的开发过程中，我们基于最先进的工作 [BALM2](https://github.com/hku-mars/BALM) 进行开发。

## **6. 许可证**
源代码以 [GPLv2](LICENSE) 许可证发布。

我们仍在努力提高代码的性能和可靠性。如有任何技术问题，请通过电子邮件 xliuaa@connect.hku.hk 联系我们。商业用途请联系张富博士 fuzhang@hku.hk。