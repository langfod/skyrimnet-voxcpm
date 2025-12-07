param(
    [switch]$test,
    [switch]$nobuild,
    [switch]$noarchive,
    [switch]$noclean
)

$PACKAGE_NAME = "SkyrimNet_VoxCPM"
$version = "1.5.0"
if (-not $nobuild -or $noclean) {
    
    if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
        Write-Host "Virtual environment not found. Please set up the virtual environment before building." -ForegroundColor Red
        exit 1
    }
    & .venv\Scripts\Activate.ps1
    if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
        & pip install pyinstaller
    }
    
    try {
        $pipListOutput = python -m pip list

        # Check if the module name is present in the output
        if ($pipListOutput -match "^pydub\s.*") {
            & pip uninstall pydub -y
            & pip uninstall pydub-ng -y
            & pip install pydub-ng
            Write-Host "Reinstalled pydub-ng to avoid conflicts."
        }
    } catch {
        Write-Warning "Python or pip might not be installed or accessible in your PATH. Error: $($_.Exception.Message)"
    }

    Write-Host "Starting build process..."
    if ($noclean) {
        Write-Host "Skipping clean step as per -noclean flag."
        & pyinstaller --noconfirm --log-level=WARN skyrimnet-voxcpm.spec
    } else {
        try {
            if (Test-Path "build") {
                Remove-Item -Path "build" -Recurse -Force
            }
            if (Test-Path "dist") {
                Remove-Item -Path "dist" -Recurse -Force
            }
            if (Test-Path "__pycache__") {
                Remove-Item -Path "__pycache__" -Recurse -Force
            }
        } catch {
            Write-Host "Error during cleanup: $_" -ForegroundColor Red
            exit 1
        }
        & pyinstaller --clean --noconfirm --log-level=ERROR skyrimnet-voxcpm.spec
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed. Exiting."
        exit $LASTEXITCODE
    }
    Deactivate
}

if ($test) {
    Write-Host "Running in test mode: Archive will be created but not deployed."
    if (-not (Test-Path "dist\skyrimnet-voxcpm\skyrimnet-voxcpm.exe")) {
        Write-Host "Error: Executable not found. Please build the project first."
        exit 1
    }
    Copy-Item -Path "models" -Destination "dist\SkyrimNet-VoxCPM\" -Force -Recurse
    Copy-Item -Path "speakers" -Destination "dist\SkyrimNet-VoxCPM\" -Force -Recurse
    Copy-Item -Path "assets" -Destination "dist\SkyrimNet-VoxCPM\" -Force -Recurse
    Copy-Item -Path "examples\Start.bat" -Destination "dist\SkyrimNet-VoxCPM\" -Force -Recurse
    Copy-Item -Path "examples\Start_VoxCPM.ps1" -Destination "dist\SkyrimNet-VoxCPM\" -Force -Recurse
    Copy-Item -Path "skyrimnet_config.txt" -Destination "dist\SkyrimNet-VoxCPM\" -Force



    Set-Location -Path ./dist/SkyrimNet-VoxCPM
    & ./Start.bat -server "localhost" -port 7860
    Set-Location -Path ../..
} else {
    Write-Host "Running in deployment mode: Archive will be created and deployed."
    if (-not (Test-Path "dist\skyrimnet-voxcpm\skyrimnet-voxcpm.exe")) {
        Write-Host "Error: Executable not found. Please build the project first."
        exit 1
    }

    if (Test-Path "archive") {
        Remove-Item -Path "archive" -Recurse -Force
    }
    New-Item -ItemType Directory -Path "archive/$PACKAGE_NAME" -Force
    New-Item -ItemType Directory -Path "archive/$PACKAGE_NAME/assets" -Force

    Get-ChildItem -Path "speakers" -Directory | Copy-Item -Destination "archive/$PACKAGE_NAME/speakers" 

    Copy-Item -Path "speakers\en\malebrute.wav" -Destination "archive/$PACKAGE_NAME/speakers/en\" -Force -Recurse
    Copy-Item -Path "speakers\en\malecommoner.wav" -Destination "archive/$PACKAGE_NAME/speakers/en\" -Force -Recurse
    Copy-Item -Path "assets\silence_100ms.wav" -Destination "archive/$PACKAGE_NAME/assets\" -Force -Recurse
    Copy-Item -Path "examples\Start.bat" -Destination "archive/$PACKAGE_NAME\" -Force
    Copy-Item -Path "examples\Start_VoxCPM.ps1" -Destination "archive/$PACKAGE_NAME\" -Force
    Copy-Item -Path "dist\skyrimnet-voxcpm\skyrimnet-voxcpm.exe" -Destination "archive/$PACKAGE_NAME\" -Force
    Copy-Item -Path "dist\skyrimnet-voxcpm\_internal" -Destination "archive/$PACKAGE_NAME\" -Force -Recurse
    Copy-Item -Path "skyrimnet_config.txt" -Destination "archive/$PACKAGE_NAME\" -Force


    if (-not $noarchive) {
        $archiveName = "$PACKAGE_NAME"

        if ($version) {
            $archiveName += "_$version"
        }
        Write-Host "Creating archive: $archiveName.zip"
        Set-Location -Path ./archive
        & "C:\Program Files\7-Zip\7z.exe" a -t7z "$archiveName.7z" "$PACKAGE_NAME" -mx=9
    }
}
