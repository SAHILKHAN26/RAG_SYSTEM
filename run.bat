@echo off
TITLE Bot Case Study Launcher

echo ==================================================
echo 🚀 Starting Bot Case Study System
echo ==================================================

REM Check if .venv exists
if not exist .venv (
    echo [ERROR] .venv not found. Please setup environment first.
    pause
    exit /b
)

echo.
echo [1/4] 📦 Initializing Database (Dropping old tables...)...
call .venv\Scripts\activate
python scripts\init_db.py --drop --seed
if %errorlevel% neq 0 (
    echo [ERROR] Database initialization failed!
    pause
    exit /b
)
echo Database initialized successfully.

echo.
echo [2/4] Launching Agents (Window 1)...
echo Starting General Agent and RAG Agent in new tabs...
start "Agents" wt new-tab -d . --title "General Agent" cmd /k "call .venv\Scripts\activate && python -m src.agents.langgraph.general_agent" ; new-tab -d . --title "RAG Agent" cmd /k "call .venv\Scripts\activate && python -m src.agents.langgraph.rag_agent"

echo.
echo ⏳ Waiting 12 seconds for agents to initialize...
timeout /t 12 /nobreak

echo.
echo [3/4] Launching Core Services (Window 2)...
echo Starting Orchestrator and Main API in new tabs...
start "Servers" wt new-tab -d . --title "Host Orchestrator" cmd /k "call .venv\Scripts\activate && python -m host.entry" ; new-tab -d . --title "FastAPI Server" cmd /k "call .venv\Scripts\activate && python main.py"

echo.
echo ✅ System Startup Initiated!
echo.
echo   - Window 1: Agents (General, RAG)
echo   - Window 2: Servers (Orchestrator, API)
echo.
echo Press any key to exit this launcher (terminals will stay open).
pause >nul
