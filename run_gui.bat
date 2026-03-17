@echo off
echo Starting Forensic Analysis GUI...

:: 1. מוודא שאנחנו בתיקייה שבה נמצא הקובץ הזה
cd /d "%~dp0"

:: 2. הפעלת הסביבה הוירטואלית
call venv_win\Scripts\activate.bat

:: 3. כניסה לתיקייה של הקוד
cd combine_features

:: 4. הרצת הסקריפט
python analyze_results.py

:: 5. השהייה בסוף כדי שאם יש שגיאה החלון לא ייסגר מיד
pause