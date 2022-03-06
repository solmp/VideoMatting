cmake_minimum_required(VERSION 3.20)
project(Demo)

set(LITE_DIR ${CMAKE_SOURCE_DIR}/lite.ai.toolkit)
set(CMAKE_CXX_STANDARD 14)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build)

set(OpenCV_LIBS
        opencv_highgui453d
        opencv_core453d
        opencv_imgcodecs453d
        opencv_imgproc453d
        opencv_video453d
        opencv_videoio453d
        )

include_directories(${LITE_DIR}/include)
link_directories(${LITE_DIR}/lib)

add_executable(Demo main.cpp)

target_link_libraries(Demo
        lite.ai.toolkit
        onnxruntime
        ${OpenCV_LIBS}
        MNN
        TNN)


