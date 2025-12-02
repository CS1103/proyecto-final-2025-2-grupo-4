````markdown
[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/o8XztwuW)
# Proyecto Final 2025-1: Celeste Neural Controller
## **CS2013 Programación III** · Informe Final

### **Descripción**

> Implementación de un **controlador de videojuegos sin contacto** basado en visión artificial y una red neuronal (MLP) construida desde cero en C++. El sistema permite jugar al videojuego *Celeste* interpretando gestos manuales capturados por cámara web en tiempo real, utilizando procesamiento de imágenes con OpenCV y aceleración por hardware con OpenMP para un rendimiento óptimo.

### Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Diseño e implementación](#2-diseño-e-implementación)
5. [Ejecución](#3-ejecución)
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)
7. [Trabajo en equipo](#5-trabajo-en-equipo)
8. [Conclusiones](#6-conclusiones)
9. [Bibliografía](#7-bibliografía)
10. [Licencia](#licencia)
---

### Datos generales

* **Tema**: Redes Neuronales (MLP) y Visión Artificial en C++
* **Grupo**: `Grupo 4`
* **Integrantes**:

  * Fabian Arana Espinoza – [20242006] (Arquitectura de la Red Neuronal y Optimización)
  * Alumno 2 – [Código] (Procesamiento de Imágenes e Integración con Juego)

---

### Requisitos e instalación

1. **Compilador**: MSVC (Visual Studio 2019 o superior) o GCC con soporte C++17.
2. **Dependencias**:
   * **CMake** 3.17+
   * **OpenCV** 4.x (Procesamiento de imágenes)
   * **OpenMP** (Paralelización de cálculos matriciales)
   
3. **Instalación y Compilación**:

   Es crítico compilar en modo **Release** para activar las optimizaciones vectoriales (AVX2) y el paralelismo.

   ```bash
   git clone [https://github.com/TU_USUARIO/celeste-neural-controller.git](https://github.com/TU_USUARIO/celeste-neural-controller.git)
   cd celeste-neural-controller
   mkdir build && cd build
   
   # Configurar en modo Release (Activa /O2 y /openmp en MSVC)
   cmake -DCMAKE_BUILD_TYPE=Release ..
   
   # Compilar
   cmake --build . --config Release
````

-----

### 1\. Investigación teórica

  * **Objetivo**: Desarrollar un Perceptrón Multicapa (MLP) eficiente capaz de clasificar gestos complejos en tiempo real con baja latencia.
  * **Fundamentos aplicados**:
    1.  **Arquitectura MLP**: Red neuronal densa configurada con:
          * Capa de Entrada: 900 neuronas (Imágenes 30x30 aplanadas).
          * Capa Oculta: 128 neuronas (Balance entre capacidad y velocidad).
          * Capa de Salida: 5 neuronas (Clases de gestos).
    2.  **Backpropagation**: Implementación manual del cálculo de gradientes para el ajuste de pesos mediante la regla de la cadena.
    3.  **Optimizador Adam**: Algoritmo de momento adaptativo utilizado para lograr una convergencia rápida y estable durante el entrenamiento.
    4.  **Funciones de Activación**:
          * **ReLU**: En capas ocultas para evitar el desvanecimiento del gradiente.
          * **Sigmoid**: En la capa de salida para obtener probabilidades normalizadas (0-1).

-----

### 2\. Diseño e implementación

#### 2.1 Arquitectura de la solución

  * **Estructura del Proyecto**:
    ```
    proyecto/
    ├── epic3/              # Librería de Red Neuronal (Header-only)
    │   ├── tensor.h        # Clase Tensor optimizada con OpenMP
    │   ├── neural_network.h # Orquestador de capas
    │   ├── nn_dense.h      # Capas densas conectadas
    │   └── ...
    ├── src/                # Código fuente de la aplicación
    │   ├── csv_generator.cpp  # Preprocesamiento del dataset HAGRID
    │   ├── trainer.cpp        # Entrenamiento del modelo y guardado de pesos
    │   ├── main_app.cpp       # Inferencia en tiempo real y control de teclado
    │   └── keyboard_controller.h # Interfaz con la API de Windows
    └── CMakeLists.txt      # Configuración de build y optimizaciones
    ```

#### 2.2 Estrategia de Control (Pantalla Dividida)

Para permitir movimientos y acciones simultáneas en *Celeste* (ej. saltar hacia la derecha), se implementó una lógica de **Split-Screen** que procesa dos ROI (Region of Interest) independientes:

  * **Mitad Izquierda (Mano Izquierda)**: Controla el **Movimiento**.

      * Gesto *Like* 👍 -\> Mover Derecha
      * Gesto *Dislike* 👎 -\> Mover Izquierda
      * Gesto *Stop* ✋ -\> Mirar Arriba
      * Gesto *Peace* ✌️ -\> Agacharse

  * **Mitad Derecha (Mano Derecha)**: Controla las **Acciones**.

      * Gesto *Fist* ✊ -\> Dash (Tecla X)
      * Gesto *Stop* ✋ -\> Saltar (Tecla C)
      * Gesto *Peace* ✌️ -\> Escalar (Tecla Z)

-----

### 3\. Ejecución

El flujo de trabajo consta de tres etapas secuenciales que deben ejecutarse en orden:

1.  **Generación de Datos**:
    Procesa el dataset HAGRID, redimensiona imágenes a 30x30 (escala de grises) y genera el archivo `celeste_dataset.csv`.
    ```bash
    ./Release/csv_generator.exe
    ```
2.  **Entrenamiento**:
    Carga el CSV, inicializa los pesos de forma aleatoria y entrena la red aprovechando todos los núcleos del CPU. Genera los archivos de pesos `.txt`.
    ```bash
    ./Release/trainer.exe
    ```
3.  **Inferencia (Juego)**:
    Abre la cámara web, divide la imagen, predice los gestos y simula las teclas virtuales en el sistema operativo.
    ```bash
    ./Release/main_app.exe
    ```

-----

### 4\. Análisis del rendimiento

  * **Entorno de Pruebas**: Procesador multinúcleo con soporte AVX2, Webcam 720p.
  * **Métricas**:
      * **Dataset**: \~3000 imágenes balanceadas (Subset de HAGRID: fist, like, dislike, stop, peace).
      * **Tiempo de entrenamiento**: Reducido drásticamente (\< 2 minutos) gracias a la implementación de **OpenMP** (`#pragma omp parallel for`) en la multiplicación de tensores y un Batch Size de 128.
      * **Uso de CPU**: \~100% durante el entrenamiento, demostrando una paralelización efectiva.
      * **Latencia de Inferencia**: \< 15ms por frame, permitiendo una experiencia de juego fluida en tiempo real.
  * **Ventajas**:
      * Independencia total de frameworks pesados de IA (PyTorch/TensorFlow).
      * Código altamente portable y optimizado.
  * **Limitaciones**:
      * Sensibilidad a condiciones de iluminación extremas.

-----

### 5\. Trabajo en equipo

| Tarea | Miembro | Rol |
| :--- | :--- | :--- |
| **Librería Core y Optimización** | Alumno 1 | Desarrollo del motor matemático (`tensor.h`), implementación de Backpropagation e integración de OpenMP. |
| **Aplicación y Visión Artificial** | Alumno 2 | Implementación de `main_app`, lógica de juego "Split-Screen", preprocesamiento de HAGRID y `keyboard_controller`. |

-----

### 6\. Conclusiones

  * **Logros**: Se logró controlar exitosamente un videojuego de alta precisión como *Celeste* utilizando únicamente una cámara web y una red neuronal implementada desde cero.
  * **Optimización**: La implementación de **OpenMP** y la compilación en modo Release fueron críticas para hacer viable el entrenamiento en CPU, reduciendo los tiempos de horas a minutos.
  * **Robustez**: El cambio de un dataset de fondo verde al dataset **HAGRID** (entornos reales) mejoró significativamente la capacidad de generalización de la red en entornos domésticos.

-----

### 7\. Bibliografía

1.  *HAGRID (HAnd Gesture Recognition Image Dataset)*. Kapitanov, A. et al. (2022). Recuperado de Kaggle.
2.  *OpenCV Documentation*. https://www.google.com/search?q=https://docs.opencv.org/
3.  *Deep Learning*. Ian Goodfellow, Yoshua Bengio and Aaron Courville. MIT Press, 2016.
4.  *OpenMP Application Programming Interface Specification*. Version 5.0. https://www.openmp.org/

-----

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](https://www.google.com/search?q=LICENSE) para detalles.

```
```