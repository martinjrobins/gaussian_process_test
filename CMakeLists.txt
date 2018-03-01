cmake_minimum_required(VERSION 2.8)

include(CMakeToolsHelpers OPTIONAL)
set( CMAKE_EXPORT_COMPILE_COMMANDS 1 )

# Boost
find_package(Boost 1.50.0 COMPONENTS python REQUIRED)
list(APPEND gptest_LIBRARIES ${Boost_LIBRARIES})
list(APPEND gptest_INCLUDES ${Boost_INCLUDE_DIRS})

# VTK 
find_package(VTK REQUIRED)
if (VTK_FOUND)
    add_definitions(-DHAVE_VTK)
endif(VTK_FOUND)
list(APPEND gptest_LIBRARIES ${VTK_LIBRARIES})
list(APPEND gptest_INCLUDES ${VTK_INCLUDE_DIRS})

# Eigen
find_package(Eigen3 REQUIRED)
add_definitions(-DHAVE_EIGEN)
list(APPEND gptest_INCLUDES ${EIGEN3_INCLUDE_DIR})

# Python
find_package(PythonLibs REQUIRED)
list(APPEND gptest_LIBRARIES ${PYTHON_LIBRARIES})
list(APPEND gptest_INCLUDES ${PYTHON_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wno-deprecated -Wno-inconsistent-missing-override -std=c++14")
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

# Aboria
set(Aboria_LOG_LEVEL 1 CACHE STRING "Logging level (1 = least, 3 = most)")
add_definitions(-DABORIA_LOG_LEVEL=${Aboria_LOG_LEVEL})
list(APPEND gptest_INCLUDES Aboria/src)
list(APPEND gptest_INCLUDES Aboria/third-party)

include_directories(${gptest_INCLUDES})

set(gptest_SOURCE
    gptest.cpp
    )

set(gptest_HEADERS
    gptest.hpp
    )

#add_executable(the_test test.cpp gptest_example.cpp)
#target_link_libraries(the_test ${gptest_LIBRARIES})

PYTHON_ADD_MODULE (gptest ${gptest_SOURCE})
target_link_libraries(gptest ${gptest_LIBRARIES})

