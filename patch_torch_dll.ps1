# Optimized binary patch using .NET methods
# OverflowError (Python int too large to convert to C long) raised in StaticCudaLauncher
# https://github.com/pytorch/pytorch/issues/162430#issuecomment-3289054096
$dllPath = ".venv\Lib\site-packages\torch\lib\torch_python.dll"
$backupPath = "$dllPath.backup"

if (-not (Test-Path $backupPath)) {
    Copy-Item $dllPath $backupPath -Force
    Write-Host "Backup created: $backupPath"
} else {
    Write-Host "Backup already exists"
}

# Read file content as string for fast search
$content = [System.IO.File]::ReadAllText($dllPath, [System.Text.Encoding]::GetEncoding("ISO-8859-1"))
$searchStr = "KiiiiisOl"
$index = $content.IndexOf($searchStr)

if ($index -ge 0) {
    Write-Host "Found at offset: $index (0x$($index.ToString('X')))"
    
    # Read as bytes, patch, write back
    $bytes = [System.IO.File]::ReadAllBytes($dllPath)
    $patchOffset = $index + $searchStr.Length - 1
    Write-Host "Current byte: 0x$($bytes[$patchOffset].ToString('X2')) ('$([char]$bytes[$patchOffset])')"
    $bytes[$patchOffset] = 0x4B  # 'K'
    Write-Host "Patching to: 0x4B ('K')"
    
    [System.IO.File]::WriteAllBytes($dllPath, $bytes)
    Write-Host "Patch applied successfully!"
    
    # Verify
    $verify = [System.IO.File]::ReadAllBytes($dllPath)
    Write-Host "Verification - byte at offset $patchOffset : 0x$($verify[$patchOffset].ToString('X2')) ('$([char]$verify[$patchOffset])')"
} else {
    Write-Host "Pattern '$searchStr' not found!"
}