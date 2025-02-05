# test/CMakeLists.txt

# Create C++ test
add_executable(flute_test flute_test.cpp)
target_link_libraries(flute_test PRIVATE flute)
add_test(NAME flute_cpp_test COMMAND flute_test)

# Python test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pyflute_test.py 
              ${CMAKE_CURRENT_BINARY_DIR}/pyflute_test.py COPYONLY)

add_test(NAME flute_python_test 
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/pyflute_test.py)
