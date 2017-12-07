cmake_minimum_required (VERSION 2.8.5)
project(sdw-qcp-o2)

# find ALPS Library
 # find_package(ALPS REQUIRED PATHS ${ALPS_ROOT_DIR} $ENV{ALPS_HOME} NO_SYSTEM_ENVIRONMENT_PATH)
 # message(STATUS "Found ALPS: ${ALPS_ROOT_DIR} (revision: ${ALPS_VERSION})")
 # include(${ALPS_USE_FILE})
 # enable_language(C CXX)

# -- HOST DISCRIMINATION --
site_name(HOST_NAME)
# some specific settings for the Cheops cluster.
# this sets CHEOPS_POSITION to -1, if not found:
string(FIND "${HOST_NAME}" "cheops" CHEOPS_POSITION)
if ("${CHEOPS_POSITION}" LESS 0)
  set(CHEOPS_FOUND "FALSE")
else ()
  message("Building on Cheops")
  set(CHEOPS_FOUND "TRUE")
endif ()
# some specific settings for the Jureca cluster.
set (FZ_JUELICH_ID_FILE "/etc/FZJ/systemname")
if (EXISTS "${FZ_JUELICH_ID_FILE}")
  file (STRINGS "${FZ_JUELICH_ID_FILE}" FZ_JUELICH_SYSTEMNAME)
  if (${FZ_JUELICH_SYSTEMNAME} STREQUAL "jureca")
    message("Building on Jureca")
    set(JURECA_FOUND "TRUE")
  else ()
    set(JURECA_FOUND "FALSE")
  endif ()
else ()
  set(JURECA_FOUND "FALSE")
endif ()
# some specific settings for the l91 work station:
string(FIND "${HOST_NAME}" "l91" L91_POSITION)
if ("${L91_POSITION}" LESS 0)
  set(L91_FOUND "FALSE")
else ()
  message("Building on l91.thp")
  set(L91_FOUND "TRUE")
endif ()
# some specific settings for the Thinkpad notebook:
string(FIND "${HOST_NAME}" "cbpro" GROUP_MACBOOK_POSITION)
if ("${GROUP_MACBOOK_POSITION}" LESS 0)
  set(GROUP_MACBOOK_FOUND "FALSE")
else ()
  message("Building on Group Macbook")
  set(GROUP_MACBOOK_FOUND "TRUE")
endif ()
# some specific settings for thp domain computers:
execute_process(COMMAND dnsdomainname OUTPUT_VARIABLE DOMAIN_NAME)
string(TOLOWER "${DOMAIN_NAME}" DOMAIN_NAME_LOWER)
string(FIND "${DOMAIN_NAME_LOWER}" "thp" THP_POSITION)
if ("${THP_POSITION}" LESS 0)
  set(THP_DOMAIN "FALSE")
else ()
  message("Building in the .thp domain")
  set(THP_DOMAIN "TRUE")
endif ()



# -- COMPILER CHOICE --

option(USE_GNU_COMPILER
  "Use gcc and g++ (default)" ON)

option(USE_INTEL_COMPILER 
  "Use Intel compiler" OFF)

option(USE_CLANG_COMPILER
  "Use the clang compiler" OFF)

if (${USE_CLANG_COMPILER})
  message("Using clang/clang++-3.4 compilers")
  set(USE_GNU_COMPILER "FALSE")
  if (${USE_INTEL_COMPILER})
    message(FATAL_ERROR "Specify either clang or Intel compilers")
  endif()
  set(CMAKE_C_COMPILER "clang-3.4")
  set(CMAKE_CXX_COMPILER "clang++-3.4")
endif ()

if (${USE_INTEL_COMPILER})
  message("Using Intel icc/icpc compilers")
  set(USE_GNU_COMPILER "FALSE")
  if (${USE_CLANG_COMPILER})
    message(FATAL_ERROR "Specify either clang or Intel compilers")
  endif()
  set(CMAKE_C_COMPILER "icc")
  set(CMAKE_CXX_COMPILER "icpc")
endif ()

if (${USE_GNU_COMPILER})
	message("Using gcc/g++ compilers")
	set(CMAKE_C_COMPILER "gcc")
	set(CMAKE_CXX_COMPILER "g++")
endif ()



# -- COMPILER AND LINKER SETTINGS --

set(OPTIMIZATION_LEVEL "-O3") 

# Cheops, Intel compiler
if (${CHEOPS_FOUND} AND ${USE_INTEL_COMPILER})
  # set(EXTRA_GCC_LOCATION_FLAGS "-gcc-name=/opt/rrzk/compiler/gnu/4.6.4/bin/gcc")
  set(EXTRA_GCC_LOCATION_FLAGS "-gcc-name=/opt/rrzk/compiler/gnu/4.8.2/bin/gcc")
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${EXTRA_GCC_LOCATION_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_GCC_LOCATION_FLAGS}")
  set(EXTRA_LIBRARY_LOCATIONS "${EXTRA_LIBRARY_LOCATIONS}" "/opt/rrzk/compiler/gnu/4.8.2/lib64")
endif ()

# CXX flags:
# -march settings
if (${USE_CLANG_COMPILER} OR NOT ${THP_DOMAIN})
  set(MARCH_FLAGS "-march=native")
else ()
  # setting to use instead of --march=native -- to be able to run on Core2 computers in the thp domain
  # use a safer setting
  # [obtained via `cc -march=native -E -v - </dev/null 2>&1 | grep cc1` on l07]
  set(MARCH_FLAGS "-march=core2 -mcx16 -msahf -mno-movbe -mno-aes -mno-pclmul -mno-popcnt -mno-abm -mno-lwp -mno-fma -mno-fma4 -mno-xop -mno-bmi -mno-tbm -mno-avx -mno-sse4.2 -msse4.1 --param l1-cache-size=32 --param l1-cache-line-size=64 --param l2-cache-size=6144 -mtune=core2 -fstack-protector")
endif ()

# set(WARNING_FLAGS "-Wall -Winit-self -Wextra -Woverloaded-virtual -Wconversion -Wno-narrowing -Wno-deprecated")
set(WARNING_FLAGS "-Wno-narrowing -Wno-deprecated")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${MARCH_FLAGS} ${WARNING_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MARCH_FLAGS} ${WARNING_FLAGS} -std=c++0x")


# - Build Type specifics:
# these are appended to ${CMAKE_C[XX]_FLAGS} for DEBUG / RELEASE / ETC builds

if (${USE_INTEL_COMPILER})
  set(EXTRA_FLAGS_DEBUG "-g -O0")
  # floating-point traps for division by zero, invalid operation, and overflow
else ()
  set(EXTRA_FLAGS_DEBUG "-g -O0")
endif ()
set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG}   ${EXTRA_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_FLAGS_DEBUG}")
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  if (${CHEOPS_FOUND} AND ${USE_INTEL_COMPILER})
  # disable GLIBCXX debug version on the cluster, where we use the Intel compiler
  # -> some code using std::map::operator[] failed to build
    add_definitions(-DMAX_DEBUG)
  else ()
    add_definitions(-D_GLIBCXX_DEBUG -DMAX_DEBUG)
  endif ()
endif ()

# Note: Let's use -g also for Release builds to make core dumps more
# readable in the debugger, we do not really need separate RelWithDebInfo
if (${USE_INTEL_COMPILER})
  set(EXTRA_FLAGS_RELEASE
    "${OPTIMIZATION_LEVEL} -ansi-alias -xSSE4.2 -no-prec-div -g")
else ()
  set(EXTRA_FLAGS_RELEASE
    "${OPTIMIZATION_LEVEL} -fno-strict-aliasing -ffast-math -g")
endif ()
set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   ${EXTRA_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_FLAGS_RELEASE}")

if(CMAKE_BUILD_TYPE MATCHES "Release")
  add_definitions(-DBOOST_DISABLE_ASSERTS -DARMA_NO_DEBUG -DNDEBUG -DHAVE_SSE2)
endif ()




# External libraries -- Armadillo
set(EXTRA_LIBRARY_LOCATIONS "${EXTRA_LIBRARY_LOCATIONS}" "$ENV{HOME}/lib" "$ENV{HOME}/opt/lib" "$ENV{HOME}/usr/lib" "$ENV{HOME}/usr/local/lib")
if (${GROUP_MACBOOK_FOUND})
  set(EXTRA_LIBRARY_LOCATIONS "${EXTRA_LIBRARY_LOCATIONS}" "/Volumes/Space/usr/lib" "/Volumes/Space/usr/local/lib")
endif ()


# if (${CHEOPS_FOUND} OR ${JURECA_FOUND})
#   set(ARMADILLO_LIBRARIES
#       "-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread")
# else ()
  # if (${L71_FOUND} OR ${L91_FOUND})
  #   set(ARMADILLO_LIBRARIES
  #     "-llapack -lblas -lgfortran")
  # else ()
  #   set(ARMADILLO_LIBRARIES "-larmadillo")
  # endif ()
  set(ARMADILLO_LIBRARIES "-lopenblas")
  # if (${THP_DOMAIN})
  #   # on thp machines: explicitly link to Lapack, not the Atlas implementation (which is not compatible to Openblas)
  #   set(EXTRA_LIBRARY_LOCATIONS "${EXTRA_LIBRARY_LOCATIONS}" "/usr/lib/lapack/")
  # else ()
  # endif ()
# endif ()


set(BOOST_LIBS "-lboost_program_options -lboost_timer -lboost_random")

link_directories(${EXTRA_LIBRARY_LOCATIONS})


# -- Excecutables --

set(dqmc_SRC montecarlo.cpp Simulation.cpp generic.cpp Parameters.cpp Logger.cpp UdV.cpp Simulation.Tests.cpp)
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ARMADILLO_LIBRARIES} ${BOOST_LIBS}")
add_executable(dqmc
  ${dqmc_SRC})
# target_link_libraries(dqmc ${ARMADILLO_LIBRARIES} ${ALPS_LIBRARIES})
target_link_libraries(dqmc ${ARMADILLO_LIBRARIES} ${BOOST_LIBS})