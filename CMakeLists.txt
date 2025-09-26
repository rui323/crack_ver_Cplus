cmake_minimum_required(VERSION 3.8)
project(cracks_detector)

# 言語標準
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

# === ONNX Runtime ===
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(ONNXRUNTIME_DIR "onnxruntime-linux-x64-1.21.0")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ONNXRUNTIME_DIR "onnxruntime-linux-aarch64-1.19.0")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# OpenCV
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc dnn)

include_directories(
  include
)

# ソースファイルを集める
file(GLOB_RECURSE NN_SOURCES "src/nn/*.cpp")
file(GLOB_RECURSE UTILS_SOURCES "src/utils/*.cpp")

# --- 実行ファイル ---
# クラック自動検出
add_executable(cracks_detection
    src/detection_main.cpp
    src/detection.cpp
    ${NN_SOURCES}
    ${UTILS_SOURCES}
)
target_link_libraries(cracks_detection
    onnxruntime
    ${OpenCV_LIBS}
)

# クラックサイズ測定
add_executable(cracks_size
    src/size_main.cpp
    src/size.cpp
    src/lsd.c
    ${NN_SOURCES}
    ${UTILS_SOURCES}
)
target_link_libraries(cracks_size
    onnxruntime
    ${OpenCV_LIBS}
)

# RPATH設定（onnxruntimeの動的ライブラリを見つける用）
set_target_properties(cracks_detection PROPERTIES 
    INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
    BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
)
set_target_properties(cracks_size PROPERTIES 
    INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
    BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
)
