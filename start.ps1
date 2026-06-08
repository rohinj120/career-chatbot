# Load .env
$envFile = Join-Path $PSScriptRoot ".env"
Get-Content $envFile | ForEach-Object {
    if ($_ -match "^\s*([^#][^=]*)=(.*)$") {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim())
    }
}

$port = $env:API_PORT
$host_ = $env:API_HOST

# Start backend
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'c:\Users\rohin jain\Desktop\career_rag_system'; uvicorn api:app --reload --host $host_ --port $port"

# Start frontend
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'c:\Users\rohin jain\Desktop\career_rag_system\career-chatbot-frontend'; npm run dev"

Write-Host "Started backend on http://${host_}:${port}"
Write-Host "Started frontend on http://localhost:5173"
