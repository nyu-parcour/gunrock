include(GoogleTest)

# metal-cpp requires one Objective-C++ translation unit with the
# NS/MTL/CA_PRIVATE_IMPLEMENTATION defines.
set(METAL_IMPL_SOURCE
  ${CMAKE_SOURCE_DIR}/include/gunrock/metal/metal_impl.mm
)

set(METAL_TEST_SOURCES
  test_device.cpp
  test_buffer.cpp
  test_compute.cpp
  test_primitives.cpp
  test_graph.cpp
  test_operators.cpp
  test_bfs.cpp
)

foreach(SOURCE IN LISTS METAL_TEST_SOURCES)
  get_filename_component(TEST_NAME ${SOURCE} NAME_WLE)
  add_executable(${TEST_NAME} ${SOURCE} ${METAL_IMPL_SOURCE})

  target_link_libraries(${TEST_NAME}
    PRIVATE essentials
    PRIVATE gtest::main
  )

  set_target_properties(${TEST_NAME}
    PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
  )

  gtest_discover_tests(${TEST_NAME})
endforeach()
