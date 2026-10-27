@echo off
rem Double-click to start the OpenCeph review workbench, then choose folders in the browser.
rem Keep this window open while reviewing; closing it stops the workbench.
rem A "UNC paths are not supported" notice above is harmless when started from a network share:
rem pushd (unlike cd) maps a temporary drive letter for the share.
pushd "%~dp0" || goto stopped
cd ..\..
python code\curation\server.py --open %*
popd
:stopped
echo.
echo The workbench has stopped. If an error is shown above, check that Python and
echo code\curation\requirements.txt are installed.
pause
