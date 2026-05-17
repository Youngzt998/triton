include(AddTritonUnitTest)

get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

add_triton_ut(
  NAME TestTritonTVMemory
  SRCS MemoryModelTest.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/../../semantics/Memory.cpp
  LIBS
    MLIRIR
    MLIRSupport
)

# Z3 linking mirrors tv/CMakeLists.txt
if(TARGET z3::libz3)
  target_link_libraries(TestTritonTVMemory PRIVATE z3::libz3)
else()
  target_include_directories(TestTritonTVMemory PRIVATE ${Z3_INCLUDE_DIRS})
  target_link_libraries(TestTritonTVMemory PRIVATE ${Z3_LIBRARIES})
endif()

# tv/ root is the include base (so #include "semantics/Memory.h" resolves)
target_include_directories(TestTritonTVMemory PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# ---------------------------------------------------------------------------
# Env (program state) tests — requires full Triton dialect stack for
# mlir::triton::PointerType and mlir::triton::FuncOp.
# Uses ${triton_libs} (same strategy as the triton-tv executable) to resolve
# the deep transitive link dependencies across all Triton dialects.
# ---------------------------------------------------------------------------

add_triton_ut(
  NAME TestTritonTVEnv
  SRCS EnvTest.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/../../semantics/Memory.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/../../semantics/Env.cpp
  LIBS
    ${triton_libs}
    MLIRIR
    MLIRSupport
)

if(TARGET z3::libz3)
  target_link_libraries(TestTritonTVEnv PRIVATE z3::libz3)
else()
  target_include_directories(TestTritonTVEnv PRIVATE ${Z3_INCLUDE_DIRS})
  target_link_libraries(TestTritonTVEnv PRIVATE ${Z3_LIBRARIES})
endif()

target_include_directories(TestTritonTVEnv PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# ---------------------------------------------------------------------------
# State (program state) tests — only needs MLIRIR (no Triton ops in tests).
# ---------------------------------------------------------------------------

add_triton_ut(
  NAME TestTritonTVState
  SRCS StateTest.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/../../semantics/Memory.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/../../semantics/Env.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/../../semantics/State.cpp
  LIBS
    ${triton_libs}
    MLIRIR
    MLIRSupport
)

if(TARGET z3::libz3)
  target_link_libraries(TestTritonTVState PRIVATE z3::libz3)
else()
  target_include_directories(TestTritonTVState PRIVATE ${Z3_INCLUDE_DIRS})
  target_link_libraries(TestTritonTVState PRIVATE ${Z3_LIBRARIES})
endif()

target_include_directories(TestTritonTVState PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
