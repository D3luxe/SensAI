@echo off
if exist "%LOCALAPPDATA%\RLBotGUIX\Python311\python.exe" (
    "%LOCALAPPDATA%\RLBotGUIX\Python311\python.exe" app.py
) else (
    py -3.11 app.py || py -3.12 app.py || python app.py
)
pause