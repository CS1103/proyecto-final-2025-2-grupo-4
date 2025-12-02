cmake_minimum_required(VERSION 3.17)
project(CelesteController VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- ESTÁNDAR DE LA INDUSTRIA: OPTIMIZACIONES ---
if(MSVC)
    # /O2: Maximizar velocidad
    # /Oi: Usar funciones intrínsecas (más rápido que llamadas normales)
    # /Ot: Priorizar velocidad sobre tamaño
    # /openmp: PERMITIR USAR TODOS LOS NÚCLEOS
    # /arch:AVX2: Usar instrucciones matemáticas modernas del procesador
    add_compile_options(/O2 /Oi /Ot /openmp /arch:AVX2)
else()
    # Versión para Linux/Mac (por si acaso)
    add_compile_options(-O3 -march=native -fopenmp)
endif()

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${CMAKE_SOURCE_DIR})

# --- Tus ejecutables ---
add_executable(csv_generator src/csv_generator.cpp)
target_link_libraries(csv_generator PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

add_executable(trainer src/trainer.cpp)
target_link_libraries(trainer PRIVATE OpenMP::OpenMP_CXX) # ¡Trainer acelerado!

add_executable(main_app src/main_app.cpp)
target_link_libraries(main_app PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

install(TARGETS csv_generator trainer main_app RUNTIME DESTINATION bin)