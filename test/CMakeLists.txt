# CMakeLists.txt
cmake_minimum_required(VERSION 3.1)
project(flute_test_project)

set(CMAKE_CXX_STANDARD 11)

# Use your actual paths
set(FLUTE_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../build")
set(FLUTE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../")

# Create executable
add_executable(flute_test src/main.cpp)

# Include Flute headers
target_include_directories(flute_test PRIVATE ${FLUTE_INCLUDE_DIR})

# Link with static library
target_link_libraries(flute_test PRIVATE ${FLUTE_LIB_DIR}/libflute.a)
