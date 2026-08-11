param(
    [int]$Port = 8501,
    [string]$Python = "python"
)

$projectRoot = Split-Path -Parent $PSScriptRoot

Start-Process `
    -FilePath $Python `
    -ArgumentList @("-m", "streamlit", "run", "evaluation/ground_truth_labeler.py", "--server.port", $Port, "--server.headless", "true") `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden
