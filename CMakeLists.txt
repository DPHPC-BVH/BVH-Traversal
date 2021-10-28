cmake_minimum_required(VERSION 3.16.0 FATAL_ERROR)

project(rt CXX)

################################################################################
# Set target arch type if empty. Visual studio solution generator provides it.
################################################################################
if(NOT CMAKE_VS_PLATFORM_NAME)
    set(CMAKE_VS_PLATFORM_NAME "x64")
endif()
message("${CMAKE_VS_PLATFORM_NAME} architecture in use")

if(NOT ("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32"
     OR "${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64"))
    message(FATAL_ERROR "${CMAKE_VS_PLATFORM_NAME} arch is not supported!")
endif()

################################################################################
# Global configuration types
################################################################################
set(CMAKE_CONFIGURATION_TYPES
    "Debug"
    "Release"
    CACHE STRING "" FORCE
)

################################################################################
# Global compiler options
################################################################################
if(MSVC)
    # remove default flags provided with CMake for MSVC
    set(CMAKE_CXX_FLAGS "")
    set(CMAKE_CXX_FLAGS_DEBUG "")
    set(CMAKE_CXX_FLAGS_RELEASE "")
endif()

################################################################################
# Global linker options
################################################################################
if(MSVC)
    # remove default flags provided with CMake for MSVC
    set(CMAKE_EXE_LINKER_FLAGS "")
    set(CMAKE_MODULE_LINKER_FLAGS "")
    set(CMAKE_SHARED_LINKER_FLAGS "")
    set(CMAKE_STATIC_LINKER_FLAGS "")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS_DEBUG "${CMAKE_MODULE_LINKER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS}")
    set(CMAKE_STATIC_LINKER_FLAGS_DEBUG "${CMAKE_STATIC_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "${CMAKE_MODULE_LINKER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS}")
    set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "${CMAKE_STATIC_LINKER_FLAGS}")
endif()

################################################################################
# Nuget packages function stub.
################################################################################
function(use_package TARGET PACKAGE VERSION)
    message(WARNING "No implementation of use_package. Create yours. "
                    "Package \"${PACKAGE}\" with version \"${VERSION}\" "
                    "for target \"${TARGET}\" is ignored!")
endfunction()

################################################################################
# Common utils
################################################################################
include(CMake/Utils.cmake)

################################################################################
# Additional Global Settings(add specific info there)
################################################################################
include(CMake/GlobalSettingsInclude.cmake OPTIONAL)

################################################################################
# Use solution folders feature
################################################################################
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Sub-projects
################################################################################



























set(PROJECT_NAME framework)

################################################################################
# Source groups
################################################################################
set(no_group_source_files
    "src/framework/base/DLLImports.inl"
)
source_group("" FILES ${no_group_source_files})

set(Headers
    "src/framework/3d/CameraControls.hpp"
    "src/framework/3d/ConvexPolyhedron.hpp"
    "src/framework/3d/Mesh.hpp"
    "src/framework/3d/Texture.hpp"
    "src/framework/3d/TextureAtlas.hpp"
    "src/framework/3rdparty/lodepng/lodepng.h"
    "src/framework/base/Array.hpp"
    "src/framework/base/BinaryHeap.hpp"
    "src/framework/base/Defs.hpp"
    "src/framework/base/Deque.hpp"
    "src/framework/base/DLLImports.hpp"
    "src/framework/base/Hash.hpp"
    "src/framework/base/Main.hpp"
    "src/framework/base/Math.hpp"
    "src/framework/base/MulticoreLauncher.hpp"
    "src/framework/base/Random.hpp"
    "src/framework/base/Sort.hpp"
    "src/framework/base/String.hpp"
    "src/framework/base/Thread.hpp"
    "src/framework/base/Timer.hpp"
    "src/framework/base/UnionFind.hpp"
    "src/framework/gpu/Buffer.hpp"
    "src/framework/gpu/CudaCompiler.hpp"
    "src/framework/gpu/CudaKernel.hpp"
    "src/framework/gpu/CudaModule.hpp"
    "src/framework/gpu/GLContext.hpp"
    "src/framework/gui/CommonControls.hpp"
    "src/framework/gui/Image.hpp"
    "src/framework/gui/Keys.hpp"
    "src/framework/gui/Window.hpp"
    "src/framework/io/AviExporter.hpp"
    "src/framework/io/File.hpp"
    "src/framework/io/ImageBinaryIO.hpp"
    "src/framework/io/ImageBmpIO.hpp"
    "src/framework/io/ImageLodePngIO.hpp"
    "src/framework/io/ImageRawPngIO.hpp"
    "src/framework/io/ImageTargaIO.hpp"
    "src/framework/io/ImageTiffIO.hpp"
    "src/framework/io/MeshBinaryIO.hpp"
    "src/framework/io/MeshWavefrontIO.hpp"
    "src/framework/io/StateDump.hpp"
    "src/framework/io/Stream.hpp"
)
source_group("Headers" FILES ${Headers})

set(Sources
    "src/framework/3d/CameraControls.cpp"
    "src/framework/3d/ConvexPolyhedron.cpp"
    "src/framework/3d/Mesh.cpp"
    "src/framework/3d/Texture.cpp"
    "src/framework/3d/TextureAtlas.cpp"
    "src/framework/3rdparty/lodepng/lodepng.cpp"
    "src/framework/base/Array.cpp"
    "src/framework/base/BinaryHeap.cpp"
    "src/framework/base/Defs.cpp"
    "src/framework/base/Deque.cpp"
    "src/framework/base/DLLImports.cpp"
    "src/framework/base/Hash.cpp"
    "src/framework/base/Main.cpp"
    "src/framework/base/Math.cpp"
    "src/framework/base/MulticoreLauncher.cpp"
    "src/framework/base/Random.cpp"
    "src/framework/base/Sort.cpp"
    "src/framework/base/String.cpp"
    "src/framework/base/Thread.cpp"
    "src/framework/base/Timer.cpp"
    "src/framework/base/UnionFind.cpp"
    "src/framework/gpu/Buffer.cpp"
    "src/framework/gpu/CudaCompiler.cpp"
    "src/framework/gpu/CudaKernel.cpp"
    "src/framework/gpu/CudaModule.cpp"
    "src/framework/gpu/GLContext.cpp"
    "src/framework/gui/CommonControls.cpp"
    "src/framework/gui/Image.cpp"
    "src/framework/gui/Keys.cpp"
    "src/framework/gui/Window.cpp"
    "src/framework/io/AviExporter.cpp"
    "src/framework/io/File.cpp"
    "src/framework/io/ImageBinaryIO.cpp"
    "src/framework/io/ImageBmpIO.cpp"
    "src/framework/io/ImageLodePngIO.cpp"
    "src/framework/io/ImageRawPngIO.cpp"
    "src/framework/io/ImageTargaIO.cpp"
    "src/framework/io/ImageTiffIO.cpp"
    "src/framework/io/MeshBinaryIO.cpp"
    "src/framework/io/MeshWavefrontIO.cpp"
    "src/framework/io/StateDump.cpp"
    "src/framework/io/Stream.cpp"
)
source_group("Sources" FILES ${Sources})

set(ALL_FILES
    ${no_group_source_files}
    ${Headers}
    ${Sources}
)

################################################################################
# Target
################################################################################
add_library(${PROJECT_NAME} STATIC ${ALL_FILES})

use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "${DEFAULT_CXX_PROPS}")
################################################################################
# Includes for CMake from *.props
################################################################################
use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "$ENV{VCTargetsPath}/BuildCustomizations/CUDA 11.4.cmake")

set(ROOT_NAMESPACE framework)

set_target_properties(${PROJECT_NAME} PROPERTIES
    VS_GLOBAL_KEYWORD "Win32Proj"
)
################################################################################
# Output directory
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        OUTPUT_DIRECTORY_DEBUG   "${CMAKE_CURRENT_SOURCE_DIR}/build/${CMAKE_VS_PLATFORM_NAME}_$<CONFIG>/${PROJECT_NAME}/"
        OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/build/${CMAKE_VS_PLATFORM_NAME}_$<CONFIG>/${PROJECT_NAME}/"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        OUTPUT_DIRECTORY_DEBUG   "${CMAKE_CURRENT_SOURCE_DIR}/build/${CMAKE_VS_PLATFORM_NAME}_$<CONFIG>/${PROJECT_NAME}/"
        OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/build/${CMAKE_VS_PLATFORM_NAME}_$<CONFIG>/${PROJECT_NAME}/"
    )
endif()
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION         "TRUE"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION         "TRUE"
    )
endif()
################################################################################
# MSVC runtime library
################################################################################
get_property(MSVC_RUNTIME_LIBRARY_DEFAULT TARGET ${PROJECT_NAME} PROPERTY MSVC_RUNTIME_LIBRARY)
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    string(CONCAT "MSVC_RUNTIME_LIBRARY_STR"
        $<$<CONFIG:Debug>:
            MultiThreadedDebug
        >
        $<$<CONFIG:Release>:
            MultiThreaded
        >
        $<$<NOT:$<OR:$<CONFIG:Debug>,$<CONFIG:Release>>>:${MSVC_RUNTIME_LIBRARY_DEFAULT}>
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    string(CONCAT "MSVC_RUNTIME_LIBRARY_STR"
        $<$<CONFIG:Debug>:
            MultiThreadedDebug
        >
        $<$<CONFIG:Release>:
            MultiThreaded
        >
        $<$<NOT:$<OR:$<CONFIG:Debug>,$<CONFIG:Release>>>:${MSVC_RUNTIME_LIBRARY_DEFAULT}>
    )
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES MSVC_RUNTIME_LIBRARY ${MSVC_RUNTIME_LIBRARY_STR})

################################################################################
# Include directories
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    target_include_directories(${PROJECT_NAME} PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src/framework;"
        "$ENV{CUDA_INC_PATH};"
        "${CMAKE_CURRENT_BINARY_DIR}\${CMAKE_CFG_INTDIR}\"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    target_include_directories(${PROJECT_NAME} PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src/framework;"
        "$ENV{CUDA_INC_PATH};"
        "${CMAKE_CURRENT_BINARY_DIR}\${CMAKE_CFG_INTDIR}\"
    )
endif()

################################################################################
# Compile definitions
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        "$<$<CONFIG:Debug>:"
            "_DEBUG"
        ">"
        "$<$<CONFIG:Release>:"
            "NDEBUG"
        ">"
        "WIN32"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        "$<$<CONFIG:Debug>:"
            "_DEBUG"
        ">"
        "$<$<CONFIG:Release>:"
            "NDEBUG"
        ">"
        "WIN32"
    )
endif()

################################################################################
# Compile and link options
################################################################################
if(MSVC)
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
        target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Debug>:
                /Od;
                /RTC1
            >
            $<$<CONFIG:Release>:
                /Ox;
                /Ot
            >
            /MP;
            /GL-;
            /Gm-;
            /W4;
            /Zi;
            /fp:fast;
            ${DEFAULT_CXX_EXCEPTION_HANDLING};
            /Y-
        )
    elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Debug>:
                /Od;
                /RTC1
            >
            $<$<CONFIG:Release>:
                /Ox;
                /Ot
            >
            /MP;
            /GL-;
            /Gm-;
            /W4;
            /Zi;
            /fp:fast;
            ${DEFAULT_CXX_EXCEPTION_HANDLING};
            /Y-
        )
    endif()
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
        string(CONCAT FILE_CL_OPTIONS
            "/W2;/wd4530"
        )
    elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        string(CONCAT FILE_CL_OPTIONS
            "/W2;/wd4530"
        )
    endif()
    source_file_compile_options(src/framework/3rdparty/lodepng/lodepng.cpp ${FILE_CL_OPTIONS})
endif()



























set(PROJECT_NAME rt)

################################################################################
# Source groups
################################################################################
set(no_group_source_files
#    "src/rt/kernels/kepler_dynamic_fetch.cu"
)
source_group("" FILES ${no_group_source_files})

set(Headers
    "src/rt/App.hpp"
    "src/rt/bvh/BVH.hpp"
    "src/rt/bvh/BVHNode.hpp"
    "src/rt/bvh/Platform.hpp"
    "src/rt/bvh/SplitBVHBuilder.hpp"
    "src/rt/cuda/CudaBVH.hpp"
    "src/rt/cuda/CudaTracer.hpp"
    "src/rt/cuda/Renderer.hpp"
    "src/rt/cuda/RendererKernels.hpp"
    "src/rt/kernels/CudaTracerKernels.hpp"
    "src/rt/ray/PixelTable.hpp"
    "src/rt/ray/RayBuffer.hpp"
    "src/rt/ray/RayBufferKernels.hpp"
    "src/rt/ray/RayGen.hpp"
    "src/rt/ray/RayGenKernels.hpp"
    "src/rt/Scene.hpp"
    "src/rt/Util.hpp"
)
source_group("Headers" FILES ${Headers})

set(Sources
    "src/rt/App.cpp"
    "src/rt/bvh/BVH.cpp"
    "src/rt/bvh/BVHNode.cpp"
    "src/rt/bvh/Platform.cpp"
    "src/rt/bvh/SplitBVHBuilder.cpp"
    "src/rt/cuda/CudaBVH.cpp"
    "src/rt/cuda/CudaTracer.cpp"
    "src/rt/cuda/Renderer.cpp"
    "src/rt/ray/PixelTable.cpp"
    "src/rt/ray/RayBuffer.cpp"
    "src/rt/ray/RayGen.cpp"
    "src/rt/Scene.cpp"
    "src/rt/Util.cpp"
)
source_group("Sources" FILES ${Sources})

set(ALL_FILES
    ${no_group_source_files}
    ${Headers}
    ${Sources}
)

################################################################################
# Target
################################################################################
add_executable(${PROJECT_NAME} ${ALL_FILES})

use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "${DEFAULT_CXX_PROPS}")
################################################################################
# Includes for CMake from *.props
################################################################################
use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "$ENV{VCTargetsPath}/BuildCustomizations/CUDA 11.4.cmake")

set(ROOT_NAMESPACE rt)

set_target_properties(${PROJECT_NAME} PROPERTIES
    VS_GLOBAL_KEYWORD "Win32Proj"
)
################################################################################
# Target name
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        TARGET_NAME_DEBUG   "${PROJECT_NAME}_${CMAKE_VS_PLATFORM_NAME}_$<CONFIG>"
        TARGET_NAME_RELEASE "${PROJECT_NAME}_${CMAKE_VS_PLATFORM_NAME}_$<CONFIG>"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        TARGET_NAME_DEBUG   "${PROJECT_NAME}_${CMAKE_VS_PLATFORM_NAME}_$<CONFIG>"
        TARGET_NAME_RELEASE "${PROJECT_NAME}_${CMAKE_VS_PLATFORM_NAME}_$<CONFIG>"
    )
endif()
################################################################################
# Output directory
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        OUTPUT_DIRECTORY_DEBUG   "${CMAKE_CURRENT_SOURCE_DIR}/"
        OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        OUTPUT_DIRECTORY_DEBUG   "${CMAKE_CURRENT_SOURCE_DIR}/"
        OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/"
    )
endif()
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION         "TRUE"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION         "TRUE"
    )
endif()
################################################################################
# MSVC runtime library
################################################################################
get_property(MSVC_RUNTIME_LIBRARY_DEFAULT TARGET ${PROJECT_NAME} PROPERTY MSVC_RUNTIME_LIBRARY)
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    string(CONCAT "MSVC_RUNTIME_LIBRARY_STR"
        $<$<CONFIG:Debug>:
            MultiThreadedDebug
        >
        $<$<CONFIG:Release>:
            MultiThreaded
        >
        $<$<NOT:$<OR:$<CONFIG:Debug>,$<CONFIG:Release>>>:${MSVC_RUNTIME_LIBRARY_DEFAULT}>
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    string(CONCAT "MSVC_RUNTIME_LIBRARY_STR"
        $<$<CONFIG:Debug>:
            MultiThreadedDebug
        >
        $<$<CONFIG:Release>:
            MultiThreaded
        >
        $<$<NOT:$<OR:$<CONFIG:Debug>,$<CONFIG:Release>>>:${MSVC_RUNTIME_LIBRARY_DEFAULT}>
    )
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES MSVC_RUNTIME_LIBRARY ${MSVC_RUNTIME_LIBRARY_STR})

################################################################################
# Include directories
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    target_include_directories(${PROJECT_NAME} PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src/rt;"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/framework;"
        "$ENV{CUDA_INC_PATH}"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    target_include_directories(${PROJECT_NAME} PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src/rt;"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/framework;"
        "$ENV{CUDA_INC_PATH}"
    )
endif()

################################################################################
# Compile definitions
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        "$<$<CONFIG:Debug>:"
            "_DEBUG"
        ">"
        "$<$<CONFIG:Release>:"
            "NDEBUG"
        ">"
        "WIN32;"
        "_CONSOLE"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        "$<$<CONFIG:Debug>:"
            "_DEBUG"
        ">"
        "$<$<CONFIG:Release>:"
            "NDEBUG"
        ">"
        "WIN32;"
        "_CONSOLE"
    )
endif()

################################################################################
# Compile and link options
################################################################################
if(MSVC)
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
        target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Debug>:
                /Od;
                /RTC1
            >
            $<$<CONFIG:Release>:
                /Ox;
                /Ot
            >
            /MP;
            /GL-;
            /Gm-;
            /W4;
            /Zi;
            /fp:fast;
            ${DEFAULT_CXX_EXCEPTION_HANDLING};
            /Y-
        )
    elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Debug>:
                /Od;
                /RTC1
            >
            $<$<CONFIG:Release>:
                /Ox;
                /Ot
            >
            /MP;
            /GL-;
            /Gm-;
            /W4;
            /Zi;
            /fp:fast;
            ${DEFAULT_CXX_EXCEPTION_HANDLING};
            /Y-
        )
    endif()
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
        target_link_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Release>:
                /OPT:REF;
                /OPT:ICF
            >
            /MANIFEST:NO;
            /DEBUG;
            /MACHINE:X86;
            /SUBSYSTEM:CONSOLE;
            /DYNAMICBASE:NO;
            /INCREMENTAL:NO
        )
    elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        target_link_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Release>:
                /OPT:REF;
                /OPT:ICF
            >
            /MANIFEST:NO;
            /DEBUG;
            /MACHINE:X64;
            /SUBSYSTEM:CONSOLE;
            /DYNAMICBASE:NO;
            /INCREMENTAL:NO
        )
    endif()
endif()

################################################################################
# Dependencies
################################################################################
# Link with other targets.
target_link_libraries(${PROJECT_NAME} PRIVATE
    framework
)

if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    set(ADDITIONAL_LIBRARY_DEPENDENCIES
        "opengl32"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    set(ADDITIONAL_LIBRARY_DEPENDENCIES
        "opengl32"
    )
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE "${ADDITIONAL_LIBRARY_DEPENDENCIES}")

