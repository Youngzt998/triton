set(Z3_ROOT "" CACHE PATH "Path to custom Z3 installation (searched first)")

# Try custom location first, then fall back to system
if(Z3_ROOT)
  find_package(Z3 4.8 CONFIG QUIET PATHS "${Z3_ROOT}" NO_DEFAULT_PATH)
  if(NOT Z3_FOUND)
    find_path(Z3_INCLUDE_DIRS NAMES z3++.h PATHS "${Z3_ROOT}/include" NO_DEFAULT_PATH)
    find_library(Z3_LIBRARIES NAMES z3 PATHS "${Z3_ROOT}/lib" NO_DEFAULT_PATH)
  endif()
endif()

# Fall back to system
if(NOT Z3_FOUND AND NOT (Z3_INCLUDE_DIRS AND Z3_LIBRARIES))
  find_package(Z3 4.8 CONFIG QUIET)
  if(NOT Z3_FOUND)
    find_path(Z3_INCLUDE_DIRS NAMES z3++.h)
    find_library(Z3_LIBRARIES NAMES z3)
  endif()
endif()

if(NOT Z3_FOUND AND NOT (Z3_INCLUDE_DIRS AND Z3_LIBRARIES))
  message(FATAL_ERROR "Z3 not found. Install libz3-dev or set -DZ3_ROOT=/path/to/z3.")
endif()

get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

add_executable(triton-tv triton-tv.cpp)

target_compile_options(triton-tv PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
target_link_libraries(triton-tv PRIVATE
  ${triton_libs}
  # tests
  TritonTestAnalysis
  TritonTestDialect
  TritonAMDGPUTestAnalysis
  TritonTestProton
  # MLIR core
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
  MLIRParser
  MLIRSupport
)

if(TARGET z3::libz3)
  target_link_libraries(triton-tv PRIVATE z3::libz3)
else()
  target_include_directories(triton-tv PRIVATE ${Z3_INCLUDE_DIRS})
  target_link_libraries(triton-tv PRIVATE ${Z3_LIBRARIES})
endif()

mlir_check_all_link_libraries(triton-tv)

add_subdirectory(test/validator)
