CMAKE_MINIMUM_REQUIRED (VERSION 3.1)
PROJECT(flute)

SET(CMAKE_CXX_STANDARD 11)
SET(FLUTE_HOME ${PROJECT_SOURCE_DIR})

# ybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.12.1  # Specify a recent stable version
)
FetchContent_MakeAvailable(pybind11)

# Original Flute library setup
set(POWV9_DAT ${FLUTE_HOME}/etc/POWV9.dat)
set(POST9_DAT ${FLUTE_HOME}/etc/POST9.dat)
set(POWV9_CPP ${CMAKE_CURRENT_BINARY_DIR}/POWV9.cpp)
set(POST9_CPP ${CMAKE_CURRENT_BINARY_DIR}/POST9.cpp)

add_custom_command(OUTPUT ${POWV9_CPP}
  COMMAND ${FLUTE_HOME}/etc/MakeDatVar.tcl powv9 ${POWV9_CPP} ${POWV9_DAT}
  DEPENDS ${POWV9_DAT} ${FLUTE_HOME}/etc/MakeDatVar.tcl
)

add_custom_command(OUTPUT ${POST9_CPP}
  COMMAND ${FLUTE_HOME}/etc/MakeDatVar.tcl post9 ${POST9_CPP} ${POST9_DAT}
  DEPENDS ${POST9_DAT} ${FLUTE_HOME}/etc/MakeDatVar.tcl
)

# Create static library
ADD_LIBRARY(flute STATIC
  flute.cpp
  ${POWV9_CPP}
  ${POST9_CPP}
)

TARGET_INCLUDE_DIRECTORIES(flute PUBLIC ${FLUTE_HOME})
set_target_properties(flute PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Python module
pybind11_add_module(pyflute python/pyflute.cpp)
target_link_libraries(pyflute PRIVATE flute)

# Add test directory
enable_testing()
add_subdirectory(tests)