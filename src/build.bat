@echo off
REM ---------------------------------------------------------------------------
REM  Build AcousticJEPA tools against the SDK bundled with the UE plugin.
REM  Run from this directory (AcousticJEPA\src). Exes go to ..\tools\.
REM ---------------------------------------------------------------------------
setlocal

set SDK=..\..\Plugins\SteamAudio\Source\SteamAudioSDK
set SA_INC=%SDK%\include
set SA_LIB=%SDK%\lib\windows-x64

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

if not exist ..\tools mkdir ..\tools

REM acoustic_env.h self-test (no jepa_dump.cpp involvement, quick sanity check)
cl /nologo /EHsc /O2 /std:c++17 /DACOUSTIC_ENV_TEST ^
   /I "%SA_INC%" ^
   /Tp acoustic_env.h ^
   /Fe:..\tools\acoustic_env_test.exe ^
   /Fo:acoustic_env_test.obj ^
   /link "%SA_LIB%\phonon.lib"
if errorlevel 1 (
    echo.
    echo BUILD FAILED ^(acoustic_env_test^)
    exit /b 1
)

REM swarm_env.h self-test (N-agent cross-view render)
cl /nologo /EHsc /O2 /std:c++17 /DSWARM_ENV_TEST ^
   /I "%SA_INC%" ^
   /Tp swarm_env.h ^
   /Fe:..\tools\swarm_env_test.exe ^
   /Fo:swarm_env_test.obj ^
   /link "%SA_LIB%\phonon.lib"
if errorlevel 1 (
    echo.
    echo BUILD FAILED ^(swarm_env_test^)
    exit /b 1
)

for %%T in (jepa_dump swarm_dump scene_dump) do (
    cl /nologo /EHsc /O2 /std:c++17 ^
       /I "%SA_INC%" ^
       %%T.cpp ^
       /Fe:..\tools\%%T.exe ^
       /link "%SA_LIB%\phonon.lib"
    if errorlevel 1 (
        echo.
        echo BUILD FAILED ^(%%T^)
        exit /b 1
    )
)

copy /Y "%SA_LIB%\phonon.dll" ..\tools\ >nul
if exist "%SA_LIB%\GPUUtilities.dll"   copy /Y "%SA_LIB%\GPUUtilities.dll" ..\tools\ >nul
if exist "%SA_LIB%\TrueAudioNext.dll"  copy /Y "%SA_LIB%\TrueAudioNext.dll" ..\tools\ >nul

echo.
echo BUILD OK: tools written to ..\tools\
endlocal
