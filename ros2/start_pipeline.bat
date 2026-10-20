@echo off
rem 在 Windows 上一键启动 ROS2 仿真: 通过 WSL 在 Ubuntu 里编译并运行, RViz2 窗口会弹到桌面上。
rem 用法: 双击运行; 或在 CMD / PowerShell 里切到本目录后带参数运行 (参数原样传给 ros2 launch):
rem     .\start_pipeline.bat scenario:=oncoming
rem     .\start_pipeline.bat controller:=pid use_ekf:=false
rem 停止: 在这个窗口按 Ctrl+C
set DISTRO=Ubuntu-24.04
wsl -d %DISTRO% --cd "%~dp0." -- bash scripts/run.sh %*
if errorlevel 1 pause
