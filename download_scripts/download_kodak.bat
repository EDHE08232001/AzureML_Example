@echo off
setlocal enabledelayedexpansion

:: Create the directory if it doesn't exist
if not exist "datasets\kodak" mkdir "datasets\kodak"

:: Loop from 1 to 24
for /L %%i in (1,1,24) do (
    set "num=0%%i"
    set "num=!num:~-2!"
    
    echo Downloading kodim!num!.png...
    curl -L "https://r0k.us/graphics/kodak/kodak/kodim!num!.png" ^
         -o "datasets\kodak\kodim!num!.png"
)

pause