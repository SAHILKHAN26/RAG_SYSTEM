@echo off

REM ==========================
REM Internal Agents
REM ==========================
wt ^
new-tab -d . --title "GENERAL Agent" cmd /k "call .venv\Scripts\activate && python -m src.agents.langgraph.general_agent" ^
; new-tab -d . --title "RAG Agent" cmd /k "call .venv\Scripts\activate && python -m src.agents.langgraph.rag_agent"

timeout /t 20 >nul

REM ==========================
REM Host Orchestrator
REM ==========================
wt ^
new-tab -d . --title "Host Orchestrator" cmd /k "call .venv\Scripts\activate && python -m host.entry"

timeout /t 10 >nul

REM ==========================
REM FastAPI Server
REM ==========================
wt ^
new-tab -d . --title "FastAPI Server" cmd /k "call .venv\Scripts\activate && python main.py"

echo ✅ All components launched successfully in Windows Terminal tabs!
