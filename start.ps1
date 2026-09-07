<#
.SYNOPSIS
  Start the kitchen assistant: checks Ollama, sets the voice backends, prints the tablet URL,
  runs the server. Double-click start.cmd or run:  .\start.ps1 [-Port 8000] [-NoVoice] [-Open]

.PARAMETER Port      HTTP/websocket port (default 8000)
.PARAMETER Model     Ollama model (default kitchen:gemma-q8; kitchen:gemma-qat is smaller, qwen-kitchen is the 27B)
.PARAMETER NoVoice   Text only: skip whisper.cpp and Kokoro (faster startup, useful for testing)
.PARAMETER Open      Open the app in the default browser once the server is up
.PARAMETER Install   Hand over to install.ps1 (full setup + logon task), then exit
.PARAMETER Dev       Development mode: auto-restart when Python files change (implies -NoVoice unless -Voice)
.PARAMETER Voice     With -Dev, keep the speech engines on (each restart re-warms them)
.PARAMETER Http      Serve plain HTTP. The tablet microphone will NOT work (browsers require HTTPS)
.PARAMETER Force     If something is already serving on the port, stop it and take over
.PARAMETER Cloud     Default. OpenRouter for inference, falling back to the local model only if
                     it errors. Needs OPENROUTER_API_KEY (setx OPENROUTER_API_KEY "sk-or-...").
.PARAMETER Local     Use the local Ollama model instead. Holds ~19 GB of VRAM for as long as
                     it stays resident, so this is no longer the default.
#>
param(
    [int]$Port = 8000,
    [string]$Model = "kitchen:gemma-q8",
    [switch]$NoVoice,
    [switch]$Open,
    [switch]$Install,
    [switch]$Dev,
    [switch]$Voice,
    [switch]$Http,
    [switch]$Cloud,
    [switch]$Local,
    [switch]$Force
)
if ($Dev -and -not $Voice) { $NoVoice = $true }

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
. (Join-Path $root 'lib.ps1')

if ($Install) {
    # install.ps1 owns setup now: it does the dependencies, certificate, firewall and task.
    $extra = @()
    if ($NoVoice) { $extra += "-NoVoice" }
    if ($Cloud) { $extra += "-Cloud" }
    & (Join-Path $root "install.ps1") -Port $Port -Model $Model @extra
    exit $LASTEXITCODE
}

$ollamaHost = if ($env:OLLAMA_HOST) { $env:OLLAMA_HOST -replace "^https?://", "" } else { "127.0.0.1:11434" }
$ollamaAddr, $ollamaPort = $ollamaHost.Split(":")
if (-not $ollamaPort) { $ollamaPort = 11434 }
$script:ollamaWhy = ""

function Test-Ollama {
    # Plain TCP probe: immune to proxy settings that Invoke-RestMethod would honour.
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect($ollamaAddr, [int]$ollamaPort, $null, $null)
        if (-not $async.AsyncWaitHandle.WaitOne(3000)) { $script:ollamaWhy = "no answer on ${ollamaAddr}:${ollamaPort} within 3 s"; return $false }
        $client.EndConnect($async)
        return $true
    } catch {
        $script:ollamaWhy = "${ollamaAddr}:${ollamaPort} -> $($_.Exception.InnerException.Message)"
        return $false
    } finally { $client.Close() }
}

# 1. Inference backend. Cloud first by default: the local model costs ~19 GB of resident
# VRAM, and it is not built, connected to or loaded until a cloud request actually fails.
if ($Local) {
    $env:COOK_LLM = "ollama"
    Write-Host "LLM: local $Model (holds VRAM while resident)"
} elseif (-not $env:OPENROUTER_API_KEY) {
    $env:COOK_LLM = "ollama"
    Write-Host "No OPENROUTER_API_KEY, so falling back to the local model $Model." -ForegroundColor Yellow
    Write-Host "  Set it once for cloud inference: setx OPENROUTER_API_KEY `"sk-or-...`"" -ForegroundColor DarkGray
} else {
    $env:COOK_LLM = "cloud"
    $cloudModel = if ($env:COOK_OPENROUTER_MODEL) { $env:COOK_OPENROUTER_MODEL } else { "google/gemini-3.8-flash" }
    Write-Host "LLM: $cloudModel via OpenRouter; local $Model only if it errors (no VRAM until then)"
}

# 2. Ollama. Required when it is doing the inference; otherwise it is only the fallback for
# when the internet drops, so a missing one is a warning rather than a reason not to cook.
# Starting the server costs no VRAM: Ollama loads a model on the first request, not at boot.
$ollamaRequired = ($env:COOK_LLM -eq "ollama")

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    if ($ollamaRequired) {
        Write-Error "The 'ollama' command is not on PATH. Install Ollama or open a new terminal after installing."; exit 1
    }
    Write-Host "Ollama is not installed: no local fallback if the cloud is unreachable." -ForegroundColor Yellow
} else {
    if (-not (Test-Ollama)) {
        # Prefer the desktop app: it owns the server, shows the tray icon, and outlives this
        # script. A bare 'ollama serve' started here would die when this console closes.
        $app = Join-Path (Split-Path (Get-Command ollama).Source) "ollama app.exe"
        $log = Join-Path $env:TEMP "ollama-serve.log"
        if (Test-Path $app) {
            Write-Host "Ollama is not listening ($script:ollamaWhy); launching the Ollama app..."
            Start-Process -FilePath $app
        } else {
            Write-Host "Ollama is not listening ($script:ollamaWhy); starting 'ollama serve'..."
            Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden -RedirectStandardError $log
        }
        $tries = 0
        while (-not (Test-Ollama) -and $tries -lt 45) { Start-Sleep -Seconds 1; $tries++ }
    }

    if (-not (Test-Ollama)) {
        if ($ollamaRequired) {
            Write-Host "Ollama did not come up ($script:ollamaWhy)."
            if ($log -and (Test-Path $log)) { Write-Host "--- ollama serve output:"; Get-Content $log -Tail 10 }
            Write-Error "Start the Ollama app from the Start menu, wait for its tray icon, then run this script again."; exit 1
        }
        Write-Host "Ollama did not start: no local fallback if the cloud is unreachable." -ForegroundColor Yellow
    } else {
        $models = (ollama list 2>&1) -join "`n"
        if ($LASTEXITCODE -ne 0) {
            if ($ollamaRequired) { Write-Error "'ollama list' failed: $models"; exit 1 }
            Write-Host "'ollama list' failed, so the local fallback may not work." -ForegroundColor Yellow
        } elseif ($models -notmatch [regex]::Escape($Model)) {
            if ($ollamaRequired) { Write-Error "Model '$Model' is not in 'ollama list'. Create it or pass -Model."; exit 1 }
            Write-Host "Ollama has no '$Model', so there is no local fallback. Create it or pass -Model." -ForegroundColor Yellow
        } elseif ($ollamaRequired) {
            Write-Host "Ollama OK, model $Model"
        } else {
            Write-Host "Ollama up, $Model available as fallback (not loaded, no VRAM used)"
        }
    }
}

# 3. GPU memory settings for Ollama. These are read by the Ollama server at startup, so they
# are set as user environment variables; if they are missing here, Ollama was started before
# they were set and should be restarted.
foreach ($pair in @(@("OLLAMA_FLASH_ATTENTION", "1"), @("OLLAMA_KV_CACHE_TYPE", "q8_0"))) {
    if ([Environment]::GetEnvironmentVariable($pair[0], "User") -ne $pair[1]) {
        [Environment]::SetEnvironmentVariable($pair[0], $pair[1], "User")
        Write-Host "Set $($pair[0])=$($pair[1]); restart the Ollama app for it to take effect."
    }
}

# 4. Voice backends
$env:COOK_MODEL = $Model
if ($NoVoice) {
    $env:COOK_STT = "none"; $env:COOK_TTS = "none"
    Write-Host "Voice off (text only)"
} else {
    $env:COOK_STT = "whisper.cpp"; $env:COOK_TTS = "kokoro"
    $vulkan = Join-Path $root "whisper.cpp\build-vulkan\bin\Release\whisper-server.exe"
    $cpu = Join-Path $root "whisper.cpp\build\bin\Release\whisper-server.exe"
    if (Test-Path $vulkan) { Write-Host "STT: whisper.cpp (Vulkan GPU build)" }
    elseif (Test-Path $cpu) { Write-Host "STT: whisper.cpp (CPU build)" }
    else { Write-Error "whisper-server.exe not found under whisper.cpp\build*; build it or use -NoVoice"; exit 1 }
    Write-Host "TTS: Kokoro"
}

# 5. Addresses. HTTPS by default: browsers only expose the microphone in a secure context,
# so a tablet on http://<lan-ip> gets no mic at all.
$scheme = if ($Http) { "http" } else { "https" }
$lan = Get-LanAddress
Write-Host ""
Write-Host "  This PC:  ${scheme}://localhost:$Port/"
if ($lan) { Write-Host "  Tablet:   ${scheme}://${lan}:$Port/" }
if ($Http) {
    Write-Host "  NOTE: plain HTTP, so the tablet microphone will not work. Drop -Http for voice."
} else {
    Write-Host "  The certificate is self-signed: accept the warning once on the tablet"
    Write-Host "  (Fully Kiosk: turn on ignoring SSL errors), or install ${scheme}://${lan}:$Port/cert"
}
Write-Host ""
# 6. Is the port already taken? A second instance is the most common way this fails, and
# retrying a bind conflict never helps, so deal with it before the supervisor loop starts.
$owner = Get-PortOwner -Port $Port
if ($owner) {
    Write-Host ""
    if ($owner.IsAssistant) {
        Write-Host "The kitchen assistant is already running on port $Port." -ForegroundColor Yellow
        Write-Host "  pid $($owner.ProcessId), started $($owner.StartTime)"
        if (-not $Force) {
            Write-Host ""
            Write-Host "  Nothing to do: open ${scheme}://localhost:$Port/ and use it."
            Write-Host "  To replace it with this one, re-run with -Force."
            Write-Host "  To restart it in place, use the gear in the tablet's top right."
            exit 0
        }
    } else {
        Write-Host "Port $Port is in use by something else:" -ForegroundColor Yellow
        Write-Host "  pid $($owner.ProcessId)  $($owner.Name)"
        if ($owner.CommandLine) { Write-Host "  $($owner.CommandLine)" }
        if (-not $Force) {
            Write-Host ""
            Write-Host "  Serve somewhere else with -Port, or stop that process first." -ForegroundColor Red
            Write-Host "  -Force stops it for you." -ForegroundColor Red
            exit 1
        }
    }
    Write-Host "-Force: stopping pid $($owner.ProcessId)..."
    Stop-Process -Id $owner.ProcessId -Force
    if (-not (Wait-PortFree -Port $Port)) {
        Write-Host "Port $Port is still held after stopping pid $($owner.ProcessId)." -ForegroundColor Red
        Write-Host "Pick another port with -Port, or reboot." -ForegroundColor Red
        exit 1
    }
    Write-Host "Port $Port is free."
}

if ($Dev) { Write-Host "Dev mode: restarts on .py changes under src\. Sessions are snapshotted, so a cook survives the restart." }
Write-Host "Starting (model + speech warm-up takes 20-40 s)... Ctrl+C stops it."

if ($Open) {
    Start-Job -ScriptBlock {
        param($u)
        for ($i = 0; $i -lt 90; $i++) {
            try {
                Invoke-RestMethod -Uri "$u/health" -TimeoutSec 2 -SkipCertificateCheck | Out-Null
                Start-Process $u; break
            } catch { Start-Sleep -Seconds 2 }
        }
    } -ArgumentList "${scheme}://localhost:$Port" | Out-Null
}

# 7. Serve, supervised.
#
# The server cannot restart itself: once it exits there is nothing left to serve the page
# that would bring it back. So this loop owns the lifecycle. The tablet's "Restart server"
# button makes the process exit with 42, which means "start me again"; exit 0 means the cook
# stopped it on purpose; anything else is a crash, which is retried with a backoff so a
# syntax error in a source file cannot spin the CPU.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$serveArgs = @("serve", "--host", "0.0.0.0", "--port", $Port, "--model", $Model)
if ($Dev) { $serveArgs += "--reload" }
if (-not $Http) { $serveArgs += "--https" }

$RESTART_EXIT = 42
$crashes = @()
while ($true) {
    uv run cooking-assistant-ai @serveArgs
    $code = $LASTEXITCODE

    if ($code -eq $RESTART_EXIT) {
        Write-Host ""
        Write-Host "--- restart requested from the tablet; starting again ---"
        Write-Host ""
        # Windows can hold the listening socket for a moment after the process goes.
        if (-not (Wait-PortFree -Port $Port)) {
            Write-Error "Port $Port did not free up after the restart; giving up rather than crash-looping."
            exit 1
        }
        $crashes = @()
        continue
    }
    if ($code -eq 0) { break }              # Ctrl+C or a clean shutdown
    if ($Dev) { Write-Host "Server exited with $code."; break }

    # Crash. Retry, but give up if it keeps failing: five times inside two minutes means
    # the code is broken, not unlucky, and a restart loop would hide the error.
    $crashes = @($crashes | Where-Object { $_ -gt (Get-Date).AddMinutes(-2) }) + (Get-Date)
    if ($crashes.Count -ge 5) {
        Write-Error "Server crashed $($crashes.Count) times in two minutes (last exit code $code). Stopping so the error is visible above."
        exit $code
    }
    $wait = [Math]::Min(30, [Math]::Pow(2, $crashes.Count))
    Write-Host "Server exited with $code; restarting in $wait s (crash $($crashes.Count) of 5)..."
    Start-Sleep -Seconds $wait
}
