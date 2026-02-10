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

mlir_check_all_link_libraries(triton-tv)
