<#
.SYNOPSIS
  One-shot setup for the kitchen assistant on Windows: dependencies, TLS certificate,
  Ollama settings, firewall rule, and a logon task so it comes up with the machine.

  Run it once:      .\install.ps1
  Local inference:  .\install.ps1 -Local     (cloud is the default)
  Check it over:    .\install.ps1 -WhatIfOnly
  Undo everything:  .\install.ps1 -Uninstall

  Everything here is idempotent: running it again after a code change is safe and is the
  supported way to repair a broken setup.

.PARAMETER Port         Port to serve on (default 8000)
.PARAMETER Model        Ollama model for local inference (default kitchen:gemma-q8)
.PARAMETER NoVoice      Skip the whisper.cpp / Kokoro checks and install text-only
.PARAMETER Local        Use the local Ollama model instead of the cloud. It holds ~19 GB of VRAM
                        while resident, which is why cloud is the default.
.PARAMETER Cloud        Kept for compatibility; cloud is now the default.
.PARAMETER NoStartup    Do everything except register the logon task
.PARAMETER NoFirewall   Skip the firewall rule (the tablet will not be able to connect from the LAN)
.PARAMETER WhatIfOnly   Report what would happen and change nothing
.PARAMETER Uninstall    Remove the logon task and the firewall rule (leaves code, venv and database alone)
#>
param(
    [int]$Port = 8000,
    [string]$Model = "kitchen:gemma-q8",
    [switch]$NoVoice,
    [switch]$Cloud,
    [switch]$Local,
    [switch]$NoStartup,
    [switch]$NoFirewall,
    [switch]$WhatIfOnly,
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
. (Join-Path $root 'lib.ps1')

$TaskName = "Kitchen Assistant"
$FirewallRule = "Kitchen Assistant ($Port)"
$script:problems = @()
$script:notes = @()

function Step($text) { Write-Host ""; Write-Host "== $text" -ForegroundColor Cyan }
function Ok($text) { Write-Host "   [ok] $text" -ForegroundColor Green }
function Warn($text) { Write-Host "   [!]  $text" -ForegroundColor Yellow; $script:notes += $text }
function Fail($text) { Write-Host "   [x]  $text" -ForegroundColor Red; $script:problems += $text }
function Would($text) { Write-Host "   [dry] would $text" -ForegroundColor DarkGray }

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

# --------------------------------------------------------------------------- uninstall

if ($Uninstall) {
    Step "Removing the logon task"
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Ok "unregistered '$TaskName'"
    } else { Ok "no task registered" }

    Step "Removing the firewall rule"
    $rule = Get-NetFirewallRule -DisplayName "Kitchen Assistant*" -ErrorAction SilentlyContinue
    if ($rule) {
        if (Test-Admin) { $rule | Remove-NetFirewallRule; Ok "removed" }
        else { Warn "needs an elevated shell; run: Remove-NetFirewallRule -DisplayName 'Kitchen Assistant*'" }
    } else { Ok "no rule present" }

    Write-Host ""
    Write-Host "Done. Your code, virtual environment and cooking.db were left untouched." -ForegroundColor Green
    exit 0
}

# --------------------------------------------------------------------------- 1. prerequisites

Write-Host "Kitchen assistant setup" -ForegroundColor White
Write-Host "  folder: $root"
if ($WhatIfOnly) { Write-Host "  DRY RUN: nothing will be changed" -ForegroundColor Yellow }

Step "Checking prerequisites"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Fail "'uv' is not on PATH. Install it, then open a new terminal and run this again:"
    Write-Host "        powershell -c `"irm https://astral.sh/uv/install.ps1 | iex`"" -ForegroundColor DarkGray
    Write-Host "        (or: winget install --id astral-sh.uv)" -ForegroundColor DarkGray
    exit 1
}
Ok "uv $((uv --version) -replace '^uv\s*','')"

$psv = $PSVersionTable.PSVersion
Ok "PowerShell $psv"

# --------------------------------------------------------------------------- 2. dependencies

Step "Installing Python dependencies"
$syncArgs = @("sync")
if (-not $NoVoice) { $syncArgs += @("--extra", "speech") }
if ($WhatIfOnly) {
    Would "run: uv $($syncArgs -join ' ')"
} else {
    # uv cannot replace files the running server has open.
    $running = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue |
               Where-Object { $_.State -eq "Running" }
    if ($running) {
        Warn "the assistant is running; stopping it so the environment can be updated"
        Stop-ScheduledTask -TaskName $TaskName
        Start-Sleep -Seconds 2
    }
    uv @syncArgs
    if ($LASTEXITCODE -ne 0) { Fail "uv sync failed (see above)"; exit 1 }
    Ok "environment ready"
}

# --------------------------------------------------------------------------- 3. inference backend

Step "Checking the inference backend"

# Cloud is the default: the local model costs ~19 GB of resident VRAM, and it is not loaded
# at all unless a cloud request fails.
$key = [Environment]::GetEnvironmentVariable("OPENROUTER_API_KEY", "User")
$haveKey = [bool]($key -or $env:OPENROUTER_API_KEY)
$useCloud = -not $Local
if ($useCloud -and -not $haveKey) {
    Warn "no OPENROUTER_API_KEY, so this will run on the local model. For cloud inference:"
    Write-Host "        setx OPENROUTER_API_KEY `"sk-or-...`"   (then re-run this script)" -ForegroundColor DarkGray
    $useCloud = $false
} elseif ($useCloud) {
    Ok "OPENROUTER_API_KEY is set; cloud inference, local model only as fallback"
    Warn "the key is read from your user environment; it is never written into this folder"
} else {
    Ok "local inference (-Local): the model stays resident in VRAM"
}

if (Get-Command ollama -ErrorAction SilentlyContinue) {
    $models = (ollama list 2>&1) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        Warn "Ollama is installed but not answering; start the Ollama app once and re-run"
    } elseif ($models -match [regex]::Escape($Model)) {
        Ok "Ollama has $Model"
    } else {
        Warn "Ollama has no model called '$Model'. Local inference will fail until you create it (pass -Model to use another)."
    }
    # Read by the Ollama server at startup, so they belong in the user environment.
    foreach ($pair in @(@("OLLAMA_FLASH_ATTENTION", "1"), @("OLLAMA_KV_CACHE_TYPE", "q8_0"))) {
        if ([Environment]::GetEnvironmentVariable($pair[0], "User") -ne $pair[1]) {
            if ($WhatIfOnly) { Would "set $($pair[0])=$($pair[1])" }
            else {
                [Environment]::SetEnvironmentVariable($pair[0], $pair[1], "User")
                Warn "set $($pair[0])=$($pair[1]) - restart the Ollama app for it to take effect"
            }
        } else { Ok "$($pair[0])=$($pair[1])" }
    }
} elseif ($useCloud) {
    Warn "Ollama is not installed, so there is no local fallback if OpenRouter is unreachable"
} else {
    Fail "Ollama is not on PATH. Install it from https://ollama.com, or drop -Local to use the cloud."
    exit 1
}

# --------------------------------------------------------------------------- 4. voice

if (-not $NoVoice) {
    Step "Checking the speech engines"
    $vulkan = Join-Path $root "whisper.cpp\build-vulkan\bin\Release\whisper-server.exe"
    $cpu = Join-Path $root "whisper.cpp\build\bin\Release\whisper-server.exe"
    if (Test-Path $vulkan) { Ok "whisper.cpp (Vulkan GPU build)" }
    elseif (Test-Path $cpu) { Ok "whisper.cpp (CPU build)" }
    else {
        Warn "whisper-server.exe not found under whisper.cpp\build*; installing text-only. Build it, then re-run."
        $NoVoice = $true
    }
}
if ($NoVoice) { Ok "voice off (text only)" }

# --------------------------------------------------------------------------- 5. TLS certificate

Step "Preparing the TLS certificate"
# The tablet's microphone only exists in a secure context, so HTTPS is not optional.
# Generating now means the first boot does not stall on it.
if ($WhatIfOnly) {
    Would "generate a self-signed certificate covering this machine's LAN addresses"
} else {
    $cert = uv run --no-sync python -c "from cooking_assistant_ai.api.tls import ensure_cert; print(ensure_cert()[0])" 2>&1
    if ($LASTEXITCODE -eq 0) { Ok "certificate at $($cert | Select-Object -Last 1)" }
    else { Warn "could not generate the certificate now; the server will do it on first start" }
}

# --------------------------------------------------------------------------- 6. database

Step "Checking the recipe database"
$db = Join-Path $root "cooking.db"
if (Test-Path $db) {
    $count = uv run --no-sync python -c "from cooking_assistant_ai.storage.db import Store; print(len(Store('cooking.db').list_recipes()))" 2>&1 | Select-Object -Last 1
    Ok "cooking.db present ($count recipes)"
} elseif ($WhatIfOnly) {
    Would "create cooking.db with the seed recipes"
} else {
    uv run --no-sync python -c "from cooking_assistant_ai.storage.db import Store; Store('cooking.db')" | Out-Null
    Ok "created cooking.db with the seed recipes"
}

# --------------------------------------------------------------------------- 7. firewall

if (-not $NoFirewall) {
    Step "Allowing the tablet through the firewall"
    $existing = Get-NetFirewallRule -DisplayName $FirewallRule -ErrorAction SilentlyContinue
    if ($existing) { Ok "rule '$FirewallRule' already exists" }
    elseif ($WhatIfOnly) { Would "add an inbound rule for TCP $Port on private networks" }
    elseif (-not (Test-Admin)) {
        Warn "not elevated, so the firewall rule was skipped. The tablet may not be able to connect."
        Write-Host "        Run this in an admin PowerShell to add it:" -ForegroundColor DarkGray
        Write-Host "        New-NetFirewallRule -DisplayName '$FirewallRule' -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port -Profile Private" -ForegroundColor DarkGray
    } else {
        # Private profile only: this should be reachable from your kitchen, not a cafe network.
        New-NetFirewallRule -DisplayName $FirewallRule -Direction Inbound -Action Allow `
            -Protocol TCP -LocalPort $Port -Profile Private | Out-Null
        Ok "inbound TCP $Port allowed on private networks"
    }
}

# --------------------------------------------------------------------------- 8. start at logon

if (-not $NoStartup) {
    Step "Registering it to start at logon"
    # A logon task, not a Windows service: Ollama runs as a tray app in your user session and
    # the GPU is only reachable from there, so a session-0 service could not talk to either.
    $ps = (Get-Command powershell.exe).Source
    $taskArgs = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Minimized -File `"$root\start.ps1`" -Port $Port -Model $Model"
    if ($NoVoice) { $taskArgs += " -NoVoice" }
    if (-not $useCloud) { $taskArgs += " -Local" }

    if ($WhatIfOnly) {
        Would "register scheduled task '$TaskName' running: powershell $taskArgs"
    } else {
        $action = New-ScheduledTaskAction -Execute $ps -Argument $taskArgs -WorkingDirectory $root
        $trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
        # Wait for the network so the LAN address is known, never time the task out, and let
        # it run on battery: a tablet in the kitchen should just find the server there.
        $settings = New-ScheduledTaskSettingsSet `
            -ExecutionTimeLimit ([TimeSpan]::Zero) `
            -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
            -StartWhenAvailable `
            -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries `
            -MultipleInstances IgnoreNew
        $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive
        Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
            -Settings $settings -Principal $principal -Force | Out-Null
        Ok "task '$TaskName' registered (starts at logon, restarts up to 3 times if it dies)"
    }
}

# --------------------------------------------------------------------------- summary

$lan = Get-LanAddress

Write-Host ""
if ($script:problems.Count) {
    Write-Host "Setup finished with problems:" -ForegroundColor Red
    $script:problems | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
} else {
    Write-Host "Setup complete." -ForegroundColor Green
}
if ($script:notes.Count) {
    Write-Host "Worth knowing:" -ForegroundColor Yellow
    $script:notes | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
}
Write-Host ""
Write-Host "  This PC:  https://localhost:$Port/"
if ($lan) { Write-Host "  Tablet:   https://${lan}:$Port/   (accept the certificate warning once)" }
Write-Host ""
if (-not $NoStartup -and -not $WhatIfOnly) {
    Write-Host "It will start automatically next time you log in. To start it now without logging out:"
    Write-Host "  Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor DarkGray
}
Write-Host "Restart it any time from the tablet: the gear in the top right -> Restart server."
Write-Host "Remove all of this with: .\install.ps1 -Uninstall"
