@echo off
REM Launches the FastAPI backend and Next.js frontend together.
REM Run this from the skin-ai root folder (double-click it, or run from cmd/PowerShell).

SET ROOT=%~dp0

REM --- Path to your venv's activate script ---
REM Your venv lives at the project root, not inside backend\, so it's
REM pointed there directly. Edit this if you move the venv later.
SET VENV_ACTIVATE="C:\Users\Nikita\skin disease detection project\venv\Scripts\activate"

echo Starting backend (FastAPI) on http://localhost:8000 ...
start "Skin AI - Backend" cmd /k "cd /d "%ROOT%backend" && call %VENV_ACTIVATE% && uvicorn main:app --reload --port 8000"

echo Waiting a few seconds for the backend to boot...
timeout /t 5 /nobreak >nul

echo Starting frontend (Next.js) on http://localhost:3000 ...
start "Skin AI - Frontend" cmd /k "cd /d "%ROOT%frontend" && npm run dev"

echo.
echo Both servers are starting in separate windows.
echo Backend:  http://localhost:8000/health
echo Frontend: http://localhost:3000
echo.
echo Close this window any time - the two server windows will keep running.
echo To stop everything, close both "Skin AI - Backend" and "Skin AI - Frontend" windows.
pause
