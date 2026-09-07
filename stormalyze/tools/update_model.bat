@echo off
REM ============================================================================
REM  Stormalyze - update the model  (Windows)
REM
REM  Drag your retrained .keras file onto this file, or double-click and paste
REM  the path when asked.
REM
REM  First run builds a private Python environment in your user folder and
REM  installs TensorFlow into it. A few minutes, roughly 250 MB.
REM  Later runs reuse it. Your training environment is never touched.
REM
REM  The `tensorflowjs` package is deliberately NOT used: it depends on
REM  tensorflow-decision-forests, which has no Windows wheels, so pip can never
REM  resolve it here. convert_model.py writes the tfjs format directly instead.
REM
REM  Requires Python 3.10, 3.11 or 3.12 from python.org.
REM  NOT the Microsoft Store build - see :nopython at the bottom for why.
REM ============================================================================

setlocal

cd /d "%~dp0.."

REM %USERPROFILE% deliberately, NOT %LOCALAPPDATA%: the Microsoft Store build of
REM Python runs in an app container that silently redirects AppData writes into
REM ...\Packages\PythonSoftwareFoundation.Python.x\LocalCache\..., so a venv created
REM there lands somewhere other than where we look for it.
set "VENV=%USERPROFILE%\.stormalyze-convert-env"
set "VPY=%VENV%\Scripts\python.exe"
set "PYLAUNCH="

echo(
echo ==================================================================
echo   Stormalyze - model update
echo ==================================================================
echo(

REM An environment can exist but be incomplete - e.g. a previous run created the
REM venv and then failed while installing. So never trust the folder alone:
REM check that the packages actually import.
if exist "%VPY%" goto :checkdeps
goto :makeenv

:checkdeps
echo Checking the conversion environment...
"%VPY%" -c "import tensorflow, tf_keras" >nul 2>&1
if %errorlevel%==0 goto :haveenv
echo Environment exists but is incomplete - finishing the install.
echo(
goto :installdeps

REM ---- find a usable interpreter (TensorFlow ships no 3.13 wheels) ------------
:makeenv
call :trypy 3.12
if defined PYLAUNCH goto :havepython
call :trypy 3.11
if defined PYLAUNCH goto :havepython
call :trypy 3.10
if defined PYLAUNCH goto :havepython
call :trybare
if defined PYLAUNCH goto :havepython
goto :nopython

:havepython
echo Using interpreter: %PYLAUNCH%
echo(
echo Creating the conversion environment. This happens once and takes a few
echo minutes - roughly 250 MB of downloads. Leave this window open.
echo(

%PYLAUNCH% -m venv "%VENV%"

if not exist "%VPY%" goto :venvfailed
"%VPY%" --version >nul 2>&1
if errorlevel 1 goto :venvfailed

"%VPY%" -m pip install --upgrade pip --quiet

REM ---- install / repair dependencies ------------------------------------------
:installdeps
echo Installing TensorFlow ^(about 250 MB, this takes a few minutes^)...
"%VPY%" -m pip install "tensorflow-cpu==2.17.1" "tf-keras==2.17.0" --quiet
if errorlevel 1 goto :pipfailed

"%VPY%" -c "import tensorflow, tf_keras" >nul 2>&1
if errorlevel 1 goto :pipfailed

echo Environment ready.
echo(

:haveenv

set "MODEL=%~1"
if not "%MODEL%"=="" goto :run

echo Drag your retrained .keras file into this window and press Enter,
echo or paste its full path.
echo(
set /p MODEL=Model file:
if "%MODEL%"=="" goto :nomodel

:run
echo(
"%VPY%" "tools\update_model.py" "%MODEL%"
echo(
pause
exit /b 0

REM ---- helpers -----------------------------------------------------------------

:trypy
py -%1 -c "import sys" >nul 2>&1
if %errorlevel%==0 set "PYLAUNCH=py -%1"
exit /b

:trybare
python -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info < (3,13) else 1)" >nul 2>&1
if %errorlevel%==0 set "PYLAUNCH=python"
exit /b

REM ---- error paths -------------------------------------------------------------

:nopython
echo No suitable Python was found.
echo(
echo This needs Python 3.10, 3.11 or 3.12. TensorFlow publishes no packages
echo for Python 3.13 yet, so a 3.13 install cannot work no matter what.
echo(
echo Install Python 3.12 from:
echo   https://www.python.org/downloads/release/python-3129/
echo(
echo During setup tick "Add python.exe to PATH".
echo(
echo Avoid the Microsoft Store build of Python. It runs sandboxed and
echo redirects writes to AppData, which breaks virtual environments in
echo confusing ways.
echo(
pause
exit /b 1

:venvfailed
echo Could not create the Python environment at:
echo   %VENV%
echo(
echo If you are using the Microsoft Store build of Python, that is the cause -
echo it sandboxes file writes. Install Python 3.12 from python.org instead:
echo   https://www.python.org/downloads/release/python-3129/
echo(
echo Otherwise, delete the folder above if it half-exists and run this again.
echo(
pause
exit /b 1

:pipfailed
echo Installing TensorFlow failed.
echo(
echo Check the Python version in the environment:
echo   "%VPY%" --version
echo TensorFlow supports 3.10 - 3.12 only.
echo(
echo To start completely fresh, delete this folder and run this file again:
echo   %VENV%
echo(
pause
exit /b 1

:nomodel
echo No model file given - nothing to do.
echo(
pause
exit /b 1
