@echo off
REM Jarvix v2 - starts the clap wake loop using the venv Python directly
REM (no 'activate' needed, so it works regardless of shell state).
cd /d C:\Users\hp\VectorDB\jarvix
".venv\Scripts\python.exe" -m app.runtime.startup
