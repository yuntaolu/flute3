# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ytlu/projects/flute3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ytlu/projects/flute3/build

# Include any dependencies generated for this target.
include CMakeFiles/pyflute.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/pyflute.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/pyflute.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pyflute.dir/flags.make

CMakeFiles/pyflute.dir/python/pyflute.cpp.o: CMakeFiles/pyflute.dir/flags.make
CMakeFiles/pyflute.dir/python/pyflute.cpp.o: /home/ytlu/projects/flute3/python/pyflute.cpp
CMakeFiles/pyflute.dir/python/pyflute.cpp.o: CMakeFiles/pyflute.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ytlu/projects/flute3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pyflute.dir/python/pyflute.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/pyflute.dir/python/pyflute.cpp.o -MF CMakeFiles/pyflute.dir/python/pyflute.cpp.o.d -o CMakeFiles/pyflute.dir/python/pyflute.cpp.o -c /home/ytlu/projects/flute3/python/pyflute.cpp

CMakeFiles/pyflute.dir/python/pyflute.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pyflute.dir/python/pyflute.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ytlu/projects/flute3/python/pyflute.cpp > CMakeFiles/pyflute.dir/python/pyflute.cpp.i

CMakeFiles/pyflute.dir/python/pyflute.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pyflute.dir/python/pyflute.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ytlu/projects/flute3/python/pyflute.cpp -o CMakeFiles/pyflute.dir/python/pyflute.cpp.s

# Object files for target pyflute
pyflute_OBJECTS = \
"CMakeFiles/pyflute.dir/python/pyflute.cpp.o"

# External object files for target pyflute
pyflute_EXTERNAL_OBJECTS =

pyflute.cpython-36m-x86_64-linux-gnu.so: CMakeFiles/pyflute.dir/python/pyflute.cpp.o
pyflute.cpython-36m-x86_64-linux-gnu.so: CMakeFiles/pyflute.dir/build.make
pyflute.cpython-36m-x86_64-linux-gnu.so: libflute.a
pyflute.cpython-36m-x86_64-linux-gnu.so: CMakeFiles/pyflute.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ytlu/projects/flute3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module pyflute.cpython-36m-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pyflute.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/strip /home/ytlu/projects/flute3/build/pyflute.cpython-36m-x86_64-linux-gnu.so

# Rule to build all files generated by this target.
CMakeFiles/pyflute.dir/build: pyflute.cpython-36m-x86_64-linux-gnu.so
.PHONY : CMakeFiles/pyflute.dir/build

CMakeFiles/pyflute.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pyflute.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pyflute.dir/clean

CMakeFiles/pyflute.dir/depend:
	cd /home/ytlu/projects/flute3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ytlu/projects/flute3 /home/ytlu/projects/flute3 /home/ytlu/projects/flute3/build /home/ytlu/projects/flute3/build /home/ytlu/projects/flute3/build/CMakeFiles/pyflute.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pyflute.dir/depend

