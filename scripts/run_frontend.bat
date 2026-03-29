@echo off
setlocal
if not exist .venv\Scripts\streamlit.exe (
	echo Streamlit not found in .venv. Run: .venv\Scripts\python.exe -m pip install -r requirements.txt
	exit /b 1
)
.venv\Scripts\streamlit.exe run frontend\streamlit_app.py
