cmake_minimum_required(VERSION 3.0.2)
project(hba)

set(CMAKE_BUILD_TYPE "Release")
ADD_COMPILE_OPTIONS(-std=c++17)
set(CMAKE_CXX_FLAGS "-std=c++17 -O3" )
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -std=c++0x -std=c++17 -fexceptions")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

# 全局定义 Eigen 版本常量，确保匹配 GTSAM 使用的 Eigen 3.3.7
add_definitions(-DEIGEN_WORLD_VERSION=3 -DEIGEN_MAJOR_VERSION=3 -DEIGEN_MINOR_VERSION=7)

find_package(PCL REQUIRED)
# 使用 GTSAM 自带的 Eigen 版本
set(EIGEN3_INCLUDE_DIR /Users/gsl/work/slam/gtsam/gtsam/3rdparty/Eigen)
include_directories(${EIGEN3_INCLUDE_DIR})
find_package(GTSAM REQUIRED QUIET)

# 查找 TBB 库
find_library(TBB_LIBRARY tbb PATHS /opt/homebrew/lib NO_DEFAULT_PATH)
find_library(TBB_MALLOC_LIBRARY tbbmalloc PATHS /opt/homebrew/lib NO_DEFAULT_PATH)
if(TBB_LIBRARY AND TBB_MALLOC_LIBRARY)
    message("Found TBB libraries: ${TBB_LIBRARY}, ${TBB_MALLOC_LIBRARY}")
    # 将 TBB 库所在目录添加到链接搜索路径
    link_directories(/opt/homebrew/lib)
else()
    message(FATAL_ERROR "TBB libraries not found")
endif()

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
)

# Hierarchical Bundle Adjustment
add_executable(hba source/hba.cpp)
# 强制设置 Eigen 版本信息，以匹配 GTSAM 的要求（Eigen 3.3.7）
target_compile_definitions(hba PRIVATE EIGEN_WORLD_VERSION=3 EIGEN_MAJOR_VERSION=3 EIGEN_MINOR_VERSION=7)
# 链接 TBB 库
target_link_libraries(hba ${PCL_LIBRARIES} gtsam /opt/homebrew/lib/libtbb.dylib /opt/homebrew/lib/libtbbmalloc.dylib)

# Remove visualize_map since it depends on ROS for visualization
# add_executable(visualize_map source/visualize.cpp)
# target_link_libraries(visualize_map ${PCL_LIBRARIES})

# Remove calculate_MME since it depends on ROS
# add_executable(calculate_MME source/calculate_MME.cpp)
# target_link_libraries(calculate_MME ${PCL_LIBRARIES} gtsam)