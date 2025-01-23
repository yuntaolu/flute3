# CMakeLists.txt
cmake_minimum_required(VERSION 3.1)
project(my_flute_project)

set(CMAKE_CXX_STANDARD 11)

# Use your actual paths
set(FLUTE_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../flute3/build")
set(FLUTE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../flute3")

# Create executable
add_executable(my_program src/main.cpp)

# Include Flute headers
target_include_directories(my_program PRIVATE ${FLUTE_INCLUDE_DIR})

# Link with static library
target_link_libraries(my_program PRIVATE ${FLUTE_LIB_DIR}/libflute.a)
