param(
    [ValidateSet("off", "status", "logon", "startup")]
    [string]$Mode = "status",
    [string]$TaskName = "FeMindEmbedService",
    [string]$LauncherPath = (Join-Path $env:USERPROFILE "femind-embed-service.cmd"),
    [string]$Distro = "Ubuntu",
    [string]$ServiceName = "femind-embed.service",
    [string]$WindowsUser = $env:USERNAME,
    [string]$WindowsPasswordEnv = "",
    [switch]$RunNow
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Write-Launcher {
    $escapedService = $ServiceName.Replace('"', '\"')
    $content = @(
        "@echo off",
        "%SystemRoot%\System32\wsl.exe -d $Distro -u root -- bash -lc ""systemctl start $escapedService >/dev/null 2>&1 || true"""
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
            throw "startup mode requires -WindowsPasswordEnv for the Windows account that owns the WSL distro"
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
        Write-Output "FeMind remote autostart is off."
        Print-Status
    }
    "status" {
        Print-Status
    }
    "logon" {
        Register-Task -TaskMode "logon"
        Write-Output "FeMind remote autostart is on (logon)."
        Print-Status
    }
    "startup" {
        Register-Task -TaskMode "startup"
        Write-Output "FeMind remote autostart is on (startup)."
        Print-Status
    }
}
