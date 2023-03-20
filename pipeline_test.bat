 
start cmd /K "C:&&cd 'C:\Users\tyfann\Documents\TU Delft\IN4325 Information Retrieval\IN4325-IR'&&python allrank\main.py --config-file-name .\allrank\config.json --run-id test_run_ranknet --job-dir .\allrank\task-data-ranknet-lr0001 "
 
start cmd /K "C:&&cd 'C:\Users\tyfann\Documents\TU Delft\IN4325 Information Retrieval\IN4325-IR'&&python test100.py "
 
start cmd /K "C:&&cd C:\Users\ldl\Desktop&&python test1000.py "
 
start cmd /K "C:&&cd C:\Users\ldl\Desktop&&python test10000.py "


@echo off

setlocal enabledelayedexpansion


set "config=allrank\config.json"
set "data_folder=data"
set "config_folder="

for /d %%f in ("%data_folder%\FOLD*_normal") do (
  set "folder=%%~nxf"
  %python% %main% !folder!
)

endlocal


@echo off

set "python=python.exe"
set "main=allrank\main.py"
set ARRAY=("approxndcg" "neuralndcg" "ranknet" "mse")

for /d %%d in ("%cd%\%data_folder%\*") do (
  set data_path=%%d
  for /r %%f in (%config_folder%\*.json) do (
    set config_path=%%f
    python allrank\main.py --config-file-name --run-id test_run_ %config_path% %config_path%
  )
)

@echo off

set DATA_DIR=data
set CONFIG_DIR=config
set ARRAY=("hello" "you" "me")

for /d %%d in (%DATA_DIR%\FOLD*_normal) do (
    set "DATA_PATH=%%d"
    for %%f in (%CONFIG_DIR%\config*.json) do (
        set "CONFIG_PATH=%%f"
        echo Running main.py with data path: %DATA_PATH% and config path: %CONFIG_PATH%
        python main.py %DATA_PATH% %CONFIG_PATH% %ARRAY%
    )
)

