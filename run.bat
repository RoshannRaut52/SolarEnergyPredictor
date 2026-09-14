@echo off
echo ============================================
echo   Solar Energy Predictor - Starting Server
echo ============================================
echo.
echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Starting Flask server...
echo.
echo Web Pages:
echo   http://localhost:5000/          - Dashboard
echo   http://localhost:5000/predict   - Predict
echo   http://localhost:5000/models    - Models
echo   http://localhost:5000/history   - History
echo   http://localhost:5000/settings  - Settings
echo   http://localhost:5000/about     - About
echo.
echo Press Ctrl+C to stop the server.
echo ============================================
echo.

python api/app.py