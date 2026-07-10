param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 5000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (
    Resolve-Path (
        Join-Path $PSScriptRoot ".."
    )
).Path

Set-Location $projectRoot

$mlflowCommand = Get-Command mlflow -ErrorAction SilentlyContinue

if ($null -eq $mlflowCommand) {
    throw (
        "The mlflow command was not found. " +
        "Activate D:\waferwatch\.venv before running this script."
    )
}

$mlflowDataDirectory = Join-Path $projectRoot "mlflow_data"
$artifactDirectory = Join-Path $mlflowDataDirectory "artifacts"

New-Item `
    -ItemType Directory `
    -Force `
    -Path $mlflowDataDirectory |
    Out-Null

New-Item `
    -ItemType Directory `
    -Force `
    -Path $artifactDirectory |
    Out-Null

$existingServer = $false

try {
    $versionResponse = Invoke-WebRequest `
        -Uri "http://${BindHost}:${Port}/version" `
        -TimeoutSec 2 `
        -UseBasicParsing

    if ($versionResponse.StatusCode -eq 200) {
        $existingServer = $true
    }
}
catch {
    $existingServer = $false
}

if ($existingServer) {
    Write-Host (
        "MLFLOW_SERVER_ALREADY_RUNNING: " +
        "http://${BindHost}:${Port}"
    )
    Write-Host (
        "Server version: " +
        $versionResponse.Content
    )
    exit 0
}

$backendStoreUri = "sqlite:///mlflow_data/mlflow.db"

$normalizedArtifactPath = (
    Resolve-Path $artifactDirectory
).Path.Replace("\", "/")

$artifactDestination = "file:///$normalizedArtifactPath"

Write-Host ""
Write-Host "============================================================"
Write-Host "Starting WaferWatch MLflow Server"
Write-Host "============================================================"
Write-Host "Project root: $projectRoot"
Write-Host "Backend store: $backendStoreUri"
Write-Host "Artifact destination: $artifactDestination"
Write-Host "Server URL: http://${BindHost}:${Port}"
Write-Host "Workers: 1"
Write-Host ""
Write-Host "Keep this terminal open."
Write-Host "Press Ctrl+C only when you want to stop MLflow."
Write-Host "============================================================"
Write-Host ""

& mlflow server `
    --backend-store-uri $backendStoreUri `
    --artifacts-destination $artifactDestination `
    --host $BindHost `
    --port $Port `
    --workers 1

if ($LASTEXITCODE -ne 0) {
    throw "MLflow server stopped with exit code $LASTEXITCODE."
}