cmake_minimum_required(VERSION 3.16)
project(minrepro LANGUAGES CXX)

# Find hip
find_package(hip REQUIRED)
find_package(rocprim REQUIRED)
find_package(GTest REQUIRED)

# Add executable
add_executable(minrepro main.cpp)

# Set C++ standard
set_target_properties(minrepro PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Link libraries
target_link_libraries(minrepro
    PRIVATE
    hip::device
    roc::rocprim
    GTest::gtest
    GTest::gtest_main
)

# Include directories
target_include_directories(minrepro
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
