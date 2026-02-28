include(FetchContent)

FetchContent_Declare(
  metal-cpp
  URL "https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(metal-cpp)

set(METAL_CPP_INCLUDE_DIR "${metal-cpp_SOURCE_DIR}" CACHE PATH "metal-cpp include directory")
message(STATUS "metal-cpp include directory: ${METAL_CPP_INCLUDE_DIR}")
