cmake_minimum_required(VERSION 3.17)
project(CelesteController VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Solo usa esto si la variable de entorno falla
set(OpenCV_DIR "C:/opencv/build") 

find_package(OpenCV REQUIRED)

# Mostrar mensaje si lo encontró
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${CMAKE_SOURCE_DIR})

add_executable(data_collector src/data_collector.cpp)
target_link_libraries(data_collector PRIVATE ${OpenCV_LIBS})

add_executable(trainer src/trainer.cpp)

add_executable(main_app src/main_app.cpp)
target_link_libraries(main_app PRIVATE ${OpenCV_LIBS})

add_executable(csv_generator src/csv_generator.cpp)
target_link_libraries(csv_generator PRIVATE ${OpenCV_LIBS})

install(TARGETS data_collector trainer main_app csv_generator RUNTIME DESTINATION bin)