@echo off
echo 프론트엔드를 pnpm으로 시작합니다...
echo.

REM pnpm이 설치되어 있는지 확인
pnpm --version >nul 2>&1
if errorlevel 1 (
    echo pnpm이 설치되어 있지 않습니다.
    echo npm을 통해 pnpm을 설치합니다...
    npm install -g pnpm
    if errorlevel 1 (
        echo pnpm 설치에 실패했습니다.
        pause
        exit /b 1
    )
)

REM frontend 디렉토리로 이동
cd frontend

REM 의존성 설치 (필요한 경우)
if not exist "node_modules" (
    echo 의존성을 설치합니다...
    pnpm install
    if errorlevel 1 (
        echo 의존성 설치에 실패했습니다.
        pause
        exit /b 1
    )
)

echo 프론트엔드 개발 서버를 시작합니다...
echo http://localhost:3000 에서 확인할 수 있습니다.
echo.

REM 개발 서버 시작
pnpm dev

pause
