CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(infer_cls CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/Classify)

set(FASTDEPLOY_INSTALL_DIR "E:\\cpp_packages\\FastDeploy\\fastdeploy-win-x64-0.0.0")
include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)
include_directories(${FASTDEPLOY_INCS})

# Only Cls
message(>>>> ${PROJECT_SOURCE_DIR})
# add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/infer_cls_api.cc)
# add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/infer_cls_openvino.cc)
add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/infer_cls_onnx_openvino.cc)
# 添加FastDeploy库依赖
target_link_libraries(${PROJECT_NAME} ${FASTDEPLOY_LIBS})
