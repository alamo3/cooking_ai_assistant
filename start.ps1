<#
.SYNOPSIS
  Start the kitchen assistant: checks Ollama, sets the voice backends, prints the tablet URL,
  runs the server. Double-click start.cmd or run:  .\start.ps1 [-Port 8000] [-NoVoice] [-Open]

.PARAMETER Port      HTTP/websocket port (default 8000)
.PARAMETER Model     Ollama model (default kitchen:gemma-q8; kitchen:gemma-qat is smaller, qwen-kitchen is the 27B)
.PARAMETER NoVoice   Text only: skip whisper.cpp and Kokoro (faster startup, useful for testing)
.PARAMETER Open      Open the app in the default browser once the server is up
.PARAMETER Install   Register a Task Scheduler job that runs this script at logon, then exit
.PARAMETER Dev       Development mode: auto-restart when Python files change (implies -NoVoice unless -Voice)
.PARAMETER Voice     With -Dev, keep the speech engines on (each restart re-warms them)
.PARAMETER Http      Serve plain HTTP. The tablet microphone will NOT work (browsers require HTTPS)
.PARAMETER Cloud     Use OpenRouter for inference, falling back to the local model if it errors.
                     Needs OPENROUTER_API_KEY set (setx OPENROUTER_API_KEY "sk-or-...").
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
    [switch]$Cloud
)
if ($Dev -and -not $Voice) { $NoVoice = $true }

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if ($Install) {
    $ps = (Get-Command powershell.exe).Source
    $taskArgs = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Minimized -File `"$root\start.ps1`" -Port $Port -Model $Model" + $(if ($NoVoice) { " -NoVoice" } else { "" })
    $action = New-ScheduledTaskAction -Execute $ps -Argument $taskArgs -WorkingDirectory $root
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
    Register-ScheduledTask -TaskName "Kitchen Assistant" -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
    Write-Host "Registered scheduled task 'Kitchen Assistant' (runs at logon). Remove with:"
    Write-Host "  Unregister-ScheduledTask -TaskName 'Kitchen Assistant' -Confirm:`$false"
    exit 0
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

# 1. Ollama
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Error "The 'ollama' command is not on PATH. Install Ollama or open a new terminal after installing."; exit 1
}
if (-not (Test-Ollama)) {
    # Prefer the desktop app: it owns the server, shows the tray icon, and outlives this script.
    # A bare 'ollama serve' started here would die when this console closes.
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
    if (-not (Test-Ollama)) {
        Write-Host "Ollama did not come up ($script:ollamaWhy)."
        if (Test-Path $log) { Write-Host "--- ollama serve output:"; Get-Content $log -Tail 10 }
        Write-Error "Start the Ollama app from the Start menu, wait for its tray icon, then run this script again."; exit 1
    }
    Write-Host "Ollama is up."
}
$models = (ollama list 2>&1) -join "`n"
if ($LASTEXITCODE -ne 0) { Write-Error "'ollama list' failed: $models"; exit 1 }
if ($models -notmatch [regex]::Escape($Model)) {
    Write-Error "Model '$Model' is not in 'ollama list'. Create it or pass -Model."; exit 1
}
Write-Host "Ollama OK, model $Model"

# 1b. GPU memory settings for Ollama. These are read by the Ollama server at startup, so they
# are set as user environment variables; if they are missing here, Ollama was started before
# they were set and should be restarted.
foreach ($pair in @(@("OLLAMA_FLASH_ATTENTION", "1"), @("OLLAMA_KV_CACHE_TYPE", "q8_0"))) {
    if ([Environment]::GetEnvironmentVariable($pair[0], "User") -ne $pair[1]) {
        [Environment]::SetEnvironmentVariable($pair[0], $pair[1], "User")
        Write-Host "Set $($pair[0])=$($pair[1]); restart the Ollama app for it to take effect."
    }
}

# 1c. Inference backend
if ($Cloud) {
    if (-not $env:OPENROUTER_API_KEY) {
        Write-Error "-Cloud needs OPENROUTER_API_KEY. Set it once with: setx OPENROUTER_API_KEY `"sk-or-...`" (then open a new terminal)."
        exit 1
    }
    $env:COOK_LLM = "cloud"
    Write-Host "LLM: OpenRouter ($(if ($env:COOK_OPENROUTER_MODEL) { $env:COOK_OPENROUTER_MODEL } else { 'openai/gpt-oss-120b' })), local $Model as fallback"
} else {
    $env:COOK_LLM = "ollama"
}

# 2. Voice backends
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

# 3. Addresses. HTTPS by default: browsers only expose the microphone in a secure context,
# so a tablet on http://<lan-ip> gets no mic at all.
$scheme = if ($Http) { "http" } else { "https" }
$lan = (Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" -and $_.PrefixOrigin -ne "WellKnown" } |
        Sort-Object InterfaceMetric | Select-Object -First 1).IPAddress
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
if ($Dev) { Write-Host "Dev mode: restarts on .py changes under src\ (the cooking session is lost on each restart)." }
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

# 4. Serve
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$serveArgs = @("serve", "--host", "0.0.0.0", "--port", $Port, "--model", $Model)
if ($Dev) { $serveArgs += "--reload" }
if (-not $Http) { $serveArgs += "--https" }
uv run cooking-assistant-ai @serveArgs
