@echo off

setlocal enabledelayedexpansion

:: 定义要执行的 Python 脚本文件
set python_script=bootstrap.py
set /A count=1

set /A total_epoch=0
for /f "tokens=1* delims=$" %%a in (params.txt) do (
    set /A total_epoch+=%%a
)

:: 循环遍历参数列表并执行 Python 脚本
for /f "tokens=1,2* delims=$" %%a in (params.txt) do (
    for /l %%p in (1,1,%%a) do (
        echo epoch[!count!/%total_epoch%]
        python %python_script% %%b
        set /A count+=1
    )
)
