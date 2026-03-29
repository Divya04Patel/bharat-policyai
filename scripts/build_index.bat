@echo off
setlocal
if not exist .venv\Scripts\python.exe (
	echo Virtual environment not found. Create it with: py -3.11 -m venv .venv
	exit /b 1
)
.venv\Scripts\python.exe -m backend.index_builder --input-dir data\raw --index-dir data\index
