get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

add_executable(tv main.cpp)

target_compile_options(tv PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
target_link_libraries(tv PRIVATE
  ${triton_libs}
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRTransforms
  MLIRParser
  MLIRSupport
)

mlir_check_all_link_libraries(tv)
