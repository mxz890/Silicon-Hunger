param(
    [ValidateSet("web", "hf", "all")]
    [string]$Mode = "web"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host ""
Write-Host "=== Silicon Hunger temporary setup ===" -ForegroundColor Cyan
Write-Host "Root: $Root"
Write-Host "Mode: $Mode"
Write-Host ""

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python not found. Install Python 3.10 first."
}

if (Get-Command py -ErrorAction SilentlyContinue) {
    $pyVersion = & py -3.10 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($pyVersion -ne "3.10") {
        throw "Python 3.10 is required, got $pyVersion."
    }
    $venvPython = Join-Path $Root "silicon_env\Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        Write-Host "[1/5] Creating venv..." -ForegroundColor Green
        & py -3.10 -m venv silicon_env
    }
}
else {
    $pyVersion = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($pyVersion -ne "3.10") {
        throw "Python 3.10 is required, got $pyVersion."
    }
    $venvPython = Join-Path $Root "silicon_env\Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        Write-Host "[1/5] Creating venv..." -ForegroundColor Green
        & python -m venv silicon_env
    }
}

if (-not (Test-Path $venvPython)) {
    throw "venv creation failed: $venvPython"
}

$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$pipIndex = "https://pypi.tuna.tsinghua.edu.cn/simple"

Write-Host "[2/5] Updating pip..." -ForegroundColor Green
& $venvPython -m pip install --index-url $pipIndex --upgrade pip

Write-Host "[3/5] Installing Python dependencies..." -ForegroundColor Green
& $venvPython -m pip install --index-url $pipIndex -r requirements_web.txt

if ($Mode -eq "hf" -or $Mode -eq "all") {
    Write-Host "[4/5] Installing Hugging Face / ModelScope dependencies..." -ForegroundColor Green
    & $venvPython -m pip install --index-url $pipIndex -r requirements_hf.txt

    Write-Host "[5/5] Downloading Qwen2.5-3B base (about 5.7GB)..." -ForegroundColor Green
    $env:HF_ENDPOINT = "https://hf-mirror.com"
    & $venvPython -c "from modelscope import snapshot_download; print(snapshot_download('qwen/Qwen2.5-3B', cache_dir='modelscope_cache'))"

    $modelPath = Join-Path $Root "modelscope_cache\models\qwen--Qwen2.5-3B\snapshots\master"
    if (Test-Path $modelPath) {
        $env:MODEL_PATH = $modelPath
        Write-Host "MODEL_PATH=$modelPath" -ForegroundColor Yellow
    }
    else {
        Write-Warning "Model path not found. Set MODEL_PATH manually if hf_state_probe.py cannot find it."
    }
}
else {
    Write-Host "[4/5] Skipping HF dependencies (Mode=web)." -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Version A (web):"
Write-Host "  ollama pull gpt-oss:20b"
Write-Host "  ollama serve"
Write-Host "  .\run_web.bat"
Write-Host ""
if ($Mode -eq "hf" -or $Mode -eq "all") {
    Write-Host "Version B (HF probes):"
    Write-Host "  .\silicon_env\Scripts\python.exe hf_state_probe.py"
    Write-Host "  .\silicon_env\Scripts\python.exe hf_steering_probe.py"
    Write-Host ""
}
Write-Host "If PowerShell blocks the venv activation, use run_web.bat." -ForegroundColor DarkGray
