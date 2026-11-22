$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Project Python is missing. Follow the environment setup instructions in README.md.'
}

& $projectPython -u (Join-Path $PSScriptRoot 'run.py') @args
exit $LASTEXITCODE
