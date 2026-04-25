@echo off
REM Air Quality Streaming - Build and Deploy Script (Windows)
REM ========================================================

echo === Air Quality Streaming - Build and Deploy ===

REM Set variables
set IMAGE_NAME=air-quality-streaming
set CONTAINER_NAME=air-quality-streaming

REM Main menu
if "%1"=="build" goto build
if "%1"=="deploy" goto deploy
if "%1"=="deploy-standalone" goto deploy_standalone
if "%1"=="status" goto status
if "%1"=="logs" goto logs
if "%1"=="stop" goto stop
if "%1"=="restart" goto restart
if "%1"=="cleanup" goto cleanup
goto usage

:build
echo 🔨 Building Docker image...
docker build -t %IMAGE_NAME%:latest .
if %errorlevel% equ 0 (
    echo ✅ Image built successfully
) else (
    echo ❌ Image build failed
    exit /b 1
)
goto end

:deploy
echo 🚀 Deploying with docker-compose...

REM Check if container is running and stop it
docker ps -q -f name=%CONTAINER_NAME% >nul 2>&1
if %errorlevel% equ 0 (
    echo 🛑 Stopping existing container...
    docker-compose down %CONTAINER_NAME%
)

REM Build and start
docker-compose up -d --build %CONTAINER_NAME%

if %errorlevel% equ 0 (
    echo ✅ Deployment successful
    echo ⏳ Waiting for container to start...
    timeout /t 10 /nobreak >nul
    call :check_status
    call :show_logs
) else (
    echo ❌ Deployment failed
    exit /b 1
)
goto end

:deploy_standalone
echo 🚀 Deploying standalone container...

REM Stop and remove existing container
docker stop %CONTAINER_NAME% >nul 2>&1
docker rm %CONTAINER_NAME% >nul 2>&1

REM Run new container
docker run -d ^
    --name %CONTAINER_NAME% ^
    --restart unless-stopped ^
    --network airflow-docker_default ^
    -e POSTGRES_HOST=postgres ^
    -e POSTGRES_DB=airflow ^
    -e POSTGRES_USER=airflow ^
    -e POSTGRES_PASSWORD=airflow ^
    -e POSTGRES_PORT=5432 ^
    -e MINIO_HOST=minio:9000 ^
    -e MINIO_ACCESS_KEY=admin ^
    -e MINIO_SECRET_KEY=admin123 ^
    -e MINIO_BUCKET=air-quality ^
    -e WAQI_TOKEN=73f88f4b0cf15fdc4984cc221d78cff444761f42 ^
    -e STREAMING_INTERVAL=300 ^
    -v "%cd%\logs:/app/logs" ^
    %IMAGE_NAME%:latest

if %errorlevel% equ 0 (
    echo ✅ Standalone deployment successful
    echo ⏳ Waiting for container to start...
    timeout /t 10 /nobreak >nul
    call :check_status
    call :show_logs
) else (
    echo ❌ Standalone deployment failed
    exit /b 1
)
goto end

:status
echo 📊 Current Status:
echo ==================
call :check_status
if %errorlevel% equ 0 (
    echo.
    echo 📈 Container Stats:
    docker stats --no-stream %CONTAINER_NAME%
    echo.
    echo 📋 Recent Logs (last 20 lines):
    docker logs --tail=20 %CONTAINER_NAME%
)
goto end

:logs
echo 📋 Showing recent logs for %CONTAINER_NAME%:
docker logs --tail=50 %CONTAINER_NAME%
goto end

:stop
echo 🛑 Stopping streaming service...
call :check_status
if %errorlevel% equ 0 (
    docker stop %CONTAINER_NAME%
    echo ✅ Service stopped
) else (
    echo ℹ️  Service was not running
)
goto end

:restart
echo 🔄 Restarting streaming service...
docker restart %CONTAINER_NAME%
if %errorlevel% equ 0 (
    echo ✅ Service restarted
    echo ⏳ Waiting for service to start...
    timeout /t 10 /nobreak >nul
    call :check_status
    call :show_logs
) else (
    echo ❌ Restart failed
    exit /b 1
)
goto end

:cleanup
echo 🧹 Cleaning up...
docker stop %CONTAINER_NAME% >nul 2>&1
docker rm %CONTAINER_NAME% >nul 2>&1
docker rmi %IMAGE_NAME%:latest >nul 2>&1
echo ✅ Cleanup completed
goto end

:check_status
docker ps -q -f name=%CONTAINER_NAME% >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Container %CONTAINER_NAME% is running
    exit /b 0
) else (
    echo ❌ Container %CONTAINER_NAME% is not running
    exit /b 1
)

:show_logs
echo 📋 Recent logs:
docker logs --tail=10 %CONTAINER_NAME%
exit /b 0

:usage
echo Usage: %0 {build^|deploy^|deploy-standalone^|status^|logs^|stop^|restart^|cleanup}
echo.
echo Commands:
echo   build             - Build Docker image
echo   deploy            - Build and deploy with docker-compose
echo   deploy-standalone - Build and deploy standalone container
echo   status            - Show service status and stats
echo   logs              - Show recent logs
echo   stop              - Stop the service
echo   restart           - Restart the service
echo   cleanup           - Stop and remove container and image
echo.
echo Examples:
echo   %0 deploy          # Deploy with docker-compose (recommended)
echo   %0 status          # Check service status
echo   %0 logs            # View recent logs
echo   %0 restart         # Restart if having issues

:end