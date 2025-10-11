@echo off

cd enhanced_iot_botscan

echo Installing minimal dependencies...
python -m pip install -r requirements-min.txt

echo Running tests...
python -m pytest tests/unit/

if %errorlevel% equ 0 (
    echo Tests passed successfully!
) else (
    echo Tests failed!
)

exit /b %errorlevel%