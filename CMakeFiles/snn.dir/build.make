# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/chenxinye/Desktop/OPENMP_SNN/snn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/chenxinye/Desktop/OPENMP_SNN/snn

# Include any dependencies generated for this target.
include CMakeFiles/snn.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/snn.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/snn.dir/flags.make

CMakeFiles/snn.dir/src/blasi.cpp.o: CMakeFiles/snn.dir/flags.make
CMakeFiles/snn.dir/src/blasi.cpp.o: src/blasi.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chenxinye/Desktop/OPENMP_SNN/snn/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/snn.dir/src/blasi.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/snn.dir/src/blasi.cpp.o -c /home/chenxinye/Desktop/OPENMP_SNN/snn/src/blasi.cpp

CMakeFiles/snn.dir/src/blasi.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/snn.dir/src/blasi.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chenxinye/Desktop/OPENMP_SNN/snn/src/blasi.cpp > CMakeFiles/snn.dir/src/blasi.cpp.i

CMakeFiles/snn.dir/src/blasi.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/snn.dir/src/blasi.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chenxinye/Desktop/OPENMP_SNN/snn/src/blasi.cpp -o CMakeFiles/snn.dir/src/blasi.cpp.s

CMakeFiles/snn.dir/src/eign.cpp.o: CMakeFiles/snn.dir/flags.make
CMakeFiles/snn.dir/src/eign.cpp.o: src/eign.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chenxinye/Desktop/OPENMP_SNN/snn/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/snn.dir/src/eign.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/snn.dir/src/eign.cpp.o -c /home/chenxinye/Desktop/OPENMP_SNN/snn/src/eign.cpp

CMakeFiles/snn.dir/src/eign.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/snn.dir/src/eign.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chenxinye/Desktop/OPENMP_SNN/snn/src/eign.cpp > CMakeFiles/snn.dir/src/eign.cpp.i

CMakeFiles/snn.dir/src/eign.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/snn.dir/src/eign.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chenxinye/Desktop/OPENMP_SNN/snn/src/eign.cpp -o CMakeFiles/snn.dir/src/eign.cpp.s

CMakeFiles/snn.dir/src/snn.cpp.o: CMakeFiles/snn.dir/flags.make
CMakeFiles/snn.dir/src/snn.cpp.o: src/snn.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chenxinye/Desktop/OPENMP_SNN/snn/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/snn.dir/src/snn.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/snn.dir/src/snn.cpp.o -c /home/chenxinye/Desktop/OPENMP_SNN/snn/src/snn.cpp

CMakeFiles/snn.dir/src/snn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/snn.dir/src/snn.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chenxinye/Desktop/OPENMP_SNN/snn/src/snn.cpp > CMakeFiles/snn.dir/src/snn.cpp.i

CMakeFiles/snn.dir/src/snn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/snn.dir/src/snn.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chenxinye/Desktop/OPENMP_SNN/snn/src/snn.cpp -o CMakeFiles/snn.dir/src/snn.cpp.s

# Object files for target snn
snn_OBJECTS = \
"CMakeFiles/snn.dir/src/blasi.cpp.o" \
"CMakeFiles/snn.dir/src/eign.cpp.o" \
"CMakeFiles/snn.dir/src/snn.cpp.o"

# External object files for target snn
snn_EXTERNAL_OBJECTS =

libsnn.a: CMakeFiles/snn.dir/src/blasi.cpp.o
libsnn.a: CMakeFiles/snn.dir/src/eign.cpp.o
libsnn.a: CMakeFiles/snn.dir/src/snn.cpp.o
libsnn.a: CMakeFiles/snn.dir/build.make
libsnn.a: CMakeFiles/snn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/chenxinye/Desktop/OPENMP_SNN/snn/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX static library libsnn.a"
	$(CMAKE_COMMAND) -P CMakeFiles/snn.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/snn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/snn.dir/build: libsnn.a

.PHONY : CMakeFiles/snn.dir/build

CMakeFiles/snn.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/snn.dir/cmake_clean.cmake
.PHONY : CMakeFiles/snn.dir/clean

CMakeFiles/snn.dir/depend:
	cd /home/chenxinye/Desktop/OPENMP_SNN/snn && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chenxinye/Desktop/OPENMP_SNN/snn /home/chenxinye/Desktop/OPENMP_SNN/snn /home/chenxinye/Desktop/OPENMP_SNN/snn /home/chenxinye/Desktop/OPENMP_SNN/snn /home/chenxinye/Desktop/OPENMP_SNN/snn/CMakeFiles/snn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/snn.dir/depend

