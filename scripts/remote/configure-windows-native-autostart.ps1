param(
    [ValidateSet("off", "status", "logon", "startup")]
    [string]$Mode = "status",
    [string]$TaskName = "FeMindEmbedServiceNative",
    [string]$LauncherPath = (Join-Path $env:USERPROFILE "run-femind-embed-native.cmd"),
    [string]$FeMindRoot = (Join-Path $env:USERPROFILE "fe-mind-winbuild"),
    [string]$CudaRoot = "",
    [string]$AuthEnvFile = (Join-Path $env:USERPROFILE ".femind-embedding-service.env"),
    [string]$Host = "127.0.0.1",
    [int]$Port = 8899,
    [string]$Prefix = "/embed",
    [string]$Device = "cuda",
    [int]$CudaOrdinal = 0,
    [string]$WindowsUser = $env:USERNAME,
    [string]$WindowsPasswordEnv = "",
    [switch]$RunNow
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Resolve-CudaRoot {
    if (-not [string]::IsNullOrWhiteSpace($CudaRoot)) {
        return $CudaRoot
    }
    if (-not [string]::IsNullOrWhiteSpace($env:CUDA_PATH)) {
        return $env:CUDA_PATH
    }
    $roots = @()
    if (Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA") {
        $roots = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory |
            Sort-Object Name -Descending |
            Select-Object -ExpandProperty FullName
    }
    if ($roots.Count -eq 0) {
        throw "could not resolve CUDA root; pass -CudaRoot explicitly"
    }
    return $roots[0]
}

function Write-Launcher {
    $resolvedCudaRoot = Resolve-CudaRoot
    $exePath = Join-Path $FeMindRoot "target\release\femind-embed-service.exe"
    $content = @(
        "@echo off",
        "call ""%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"" >nul 2>&1",
        "set CL=/Zc:preprocessor",
        "set CUDA_PATH=$resolvedCudaRoot",
        "set CUDA_ROOT=$resolvedCudaRoot",
        "set CUDA_TOOLKIT_ROOT_DIR=$resolvedCudaRoot",
        "set PATH=%CUDA_PATH%\bin;%USERPROFILE%\.cargo\bin;%PATH%",
        "set LIB=%CUDA_PATH%\lib\x64;%LIB%",
        "cd /d ""$FeMindRoot""",
        """$exePath"" serve --host $Host --port $Port --prefix $Prefix --device $Device --cuda-ordinal $CudaOrdinal --auth-token-env-file ""$AuthEnvFile"" 1>>""%USERPROFILE%\femind-embed-service.log"" 2>>""%USERPROFILE%\femind-embed-service.err.log"""
    )
    Set-Content -Path $LauncherPath -Value $content -Encoding Ascii
}

function Get-ExistingTask {
    try {
        Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    }
    catch {
        $null
    }
}

function Remove-Launcher {
    if (Test-Path $LauncherPath) {
        Remove-Item $LauncherPath -Force
    }
}

function Remove-Task {
    if (Get-ExistingTask) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false | Out-Null
    }
}

function Print-Status {
    if (Get-ExistingTask) {
        Write-Output "autostart: on"
        schtasks.exe /Query /TN $TaskName /V /FO LIST
    }
    else {
        Write-Output "autostart: off"
    }

    if (Test-Path $LauncherPath) {
        Write-Output "--- launcher ---"
        Get-Content $LauncherPath
    }
}

function Register-Task {
    param(
        [ValidateSet("logon", "startup")]
        [string]$TaskMode
    )

    Write-Launcher
    Remove-Task

    $taskCommand = "cmd.exe /c `"$LauncherPath`""

    if ($TaskMode -eq "startup") {
        if ([string]::IsNullOrWhiteSpace($WindowsPasswordEnv)) {
            throw "startup mode requires -WindowsPasswordEnv for the Windows account"
        }
        $windowsPassword = [Environment]::GetEnvironmentVariable($WindowsPasswordEnv)
        if ([string]::IsNullOrWhiteSpace($windowsPassword)) {
            throw "startup mode could not read a Windows password from environment variable '$WindowsPasswordEnv'"
        }
        schtasks.exe /Create /F /SC ONSTART /TN $TaskName /TR $taskCommand /RU $WindowsUser /RP $windowsPassword /RL HIGHEST *> $null
    }
    else {
        schtasks.exe /Create /F /SC ONLOGON /TN $TaskName /TR $taskCommand /RL HIGHEST *> $null
    }

    if ($RunNow) {
        schtasks.exe /Run /TN $TaskName *> $null
    }
}

switch ($Mode) {
    "off" {
        Remove-Task
        Remove-Launcher
        Write-Output "FeMind native Windows autostart is off."
        Print-Status
    }
    "status" {
        Print-Status
    }
    "logon" {
        Register-Task -TaskMode "logon"
        Write-Output "FeMind native Windows autostart is on (logon)."
        Print-Status
    }
    "startup" {
        Register-Task -TaskMode "startup"
        Write-Output "FeMind native Windows autostart is on (startup)."
        Print-Status
    }
}
