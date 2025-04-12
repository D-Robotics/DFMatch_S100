cmake_minimum_required(VERSION 3.10)
project(DFMatch_S100)

set(CMAKE_CXX_STANDARD 17)

# Release 模式优化
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wall")

# Debug 模式保留调试信息
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g  -Wall")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

link_directories(
  /usr/hobot/lib/
  /usr/lib/aarch64-linux-gnu/
)

add_executable(DFMatch_S100 DFMatch.cpp)
target_link_libraries(DFMatch_S100 dnn hbucp ${OpenCV_LIBS})

