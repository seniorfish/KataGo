@echo off
setlocal EnableDelayedExpansion

rem ============================================================================
rem build_wasm_protobuf.bat - build protobuf v3.21.12 for the Emscripten (WASM) build
rem
rem The WASM ONNX build compiles neuralnet/onnxmodelbuilder.cpp, which needs the generated
rem onnx.pb.h plus a libprotobuf compiled for wasm32. This script produces both:
rem   - a native (host) protoc.exe  (used at CMake configure time to regenerate onnx.pb.*)
rem   - an emcc-compiled libprotobuf.a install tree  (passed as KATAGO_WASM_PROTOBUF_ROOT)
rem
rem Usage:
rem   build_wasm_protobuf.bat [protobuf-src-dir] [install-prefix] [protoc.exe]
rem     protobuf-src-dir : where protobuf v3.21.12 lives (downloaded here if missing)
rem     install-prefix   : where the wasm libprotobuf is installed (KATAGO_WASM_PROTOBUF_ROOT)
rem     protoc.exe       : native protoc to use; if absent, one is built from protobuf-src-dir
rem
rem Requires an activated Emscripten SDK (emcmake on PATH) and a working native C++ toolchain
rem for the host protoc build (e.g. an MSVC developer prompt via vcvars64).
rem ============================================================================

set "SCRIPT_DIR=%~dp0"
set "DEPS_DIR=%SCRIPT_DIR%..\build-wasm-deps"

set "PB_SRC=%~1"
if "%PB_SRC%"=="" set "PB_SRC=%DEPS_DIR%\protobuf-3.21.12"
set "PB_INSTALL=%~2"
if "%PB_INSTALL%"=="" set "PB_INSTALL=%DEPS_DIR%\protobuf-install"
set "PB_PROTOC=%~3"

set "PB_VERSION=v3.21.12"
set "PB_TARBALL=%DEPS_DIR%\protobuf-3.21.12.tar.gz"
set "PB_URL=https://codeload.github.com/protocolbuffers/protobuf/tar.gz/refs/tags/%PB_VERSION%"

rem ---- 1. obtain the source ----
if not exist "%PB_SRC%\CMakeLists.txt" (
  if not exist "%DEPS_DIR%" mkdir "%DEPS_DIR%"
  if not exist "%PB_TARBALL%" (
    echo Downloading protobuf %PB_VERSION% ...
    curl -L -o "%PB_TARBALL%" "%PB_URL%" || (echo ERROR: failed to download protobuf & exit /b 1)
  )
  echo Extracting protobuf %PB_VERSION% ...
  if exist "%PB_SRC%" rmdir /s /q "%PB_SRC%"
  rem Use the system bsdtar explicitly: GNU tar (e.g. from Git Bash) mis-parses Windows drive-letter
  rem paths like "D:\..." as remote hosts ("Cannot connect to D:").
  %SystemRoot%\System32\tar.exe -xzf "%PB_TARBALL%" -C "%DEPS_DIR%" || (echo ERROR: failed to extract protobuf & exit /b 1)
)

rem ---- 2. native host protoc ----
if "%PB_PROTOC%"=="" (
  set "PB_PROTOC="
  if exist "%DEPS_DIR%\protoc-build\protoc.exe" set "PB_PROTOC=%DEPS_DIR%\protoc-build\protoc.exe"
  if exist "%DEPS_DIR%\protoc-build\Release\protoc.exe" set "PB_PROTOC=%DEPS_DIR%\protoc-build\Release\protoc.exe"
  if not exist "!PB_PROTOC!" (
    echo Building native host protoc ...
    cmake -S "%PB_SRC%" -B "%DEPS_DIR%\protoc-build" -DCMAKE_BUILD_TYPE=Release -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF -Dprotobuf_WITH_ZLIB=OFF -Dprotobuf_BUILD_PROTOC_BINARIES=ON || exit /b 1
    cmake --build "%DEPS_DIR%\protoc-build" --target protoc --config Release -j %NUMBER_OF_PROCESSORS% || exit /b 1
    if exist "%DEPS_DIR%\protoc-build\protoc.exe" set "PB_PROTOC=%DEPS_DIR%\protoc-build\protoc.exe"
    if exist "%DEPS_DIR%\protoc-build\Release\protoc.exe" set "PB_PROTOC=%DEPS_DIR%\protoc-build\Release\protoc.exe"
  )
)
if "%PB_PROTOC%"=="" (echo ERROR: protoc not found after the host build & exit /b 1)
if not exist "%PB_PROTOC%" (echo ERROR: protoc not found at "%PB_PROTOC%" & exit /b 1)

rem ---- 3. wasm libprotobuf ----
echo Building wasm libprotobuf (this takes a while; parallel with -j) ...
rem -pthread is required: the engine links with --shared-memory, so every object must carry the
rem atomics/bulk-memory features or wasm-ld rejects it ("--shared-memory is disallowed by ...").
emcmake cmake -S "%PB_SRC%" -B "%DEPS_DIR%\protobuf-wasm" -DCMAKE_BUILD_TYPE=Release -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF -Dprotobuf_BUILD_PROTOC_BINARIES=OFF -Dprotobuf_WITH_ZLIB=OFF -DCMAKE_INSTALL_PREFIX="%PB_INSTALL%" -DCMAKE_C_FLAGS=-pthread -DCMAKE_CXX_FLAGS=-pthread || exit /b 1
rem Build both runtime libs - cmake --install expects libprotobuf-lite.a as well.
cmake --build "%DEPS_DIR%\protobuf-wasm" --target libprotobuf libprotobuf-lite --config Release -j %NUMBER_OF_PROCESSORS% || exit /b 1
cmake --install "%DEPS_DIR%\protobuf-wasm" || exit /b 1

rem ---- 4. report ----
echo.
echo ============================================================
echo protobuf for WASM ready:
echo   KATAGO_WASM_PROTOBUF_ROOT = %PB_INSTALL%
echo   KATAGO_WASM_PROTOC        = %PB_PROTOC%
echo.
echo Configure the KataGo WASM build with these, e.g.:
echo   emcmake cmake -S KataGo/cpp -B KataGo/cpp/build-wasm -DUSE_BACKEND=ONNX ^
echo     -DCMAKE_BUILD_TYPE=Release -DKATAGO_WASM_PROTOBUF_ROOT="%PB_INSTALL%" ^
echo     -DKATAGO_WASM_PROTOC="%PB_PROTOC%"
echo ============================================================
exit /b 0
