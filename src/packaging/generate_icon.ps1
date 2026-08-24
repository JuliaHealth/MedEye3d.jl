# Generates a medical imaging themed icon (.ico) for MedEye3D with multiple resolution layers (16x16, 32x32, 48x48, 64x64, 128x128, 256x256)
Add-Type -AssemblyName System.Drawing

$outputPath = Join-Path $PSScriptRoot "app_icon.ico"
$sizes = @(16, 32, 48, 64, 128, 256)
$bitmaps = @()

foreach ($sz in $sizes) {
    $bmp = New-Object System.Drawing.Bitmap($sz, $sz, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $g.Clear([System.Drawing.Color]::Transparent)

    # Background rounded container / medical badge
    $rect = New-Object System.Drawing.Rectangle(1, 1, ($sz - 2), ($sz - 2))
    $c1 = [System.Drawing.Color]::FromArgb(255, 12, 28, 60)
    $c2 = [System.Drawing.Color]::FromArgb(255, 20, 75, 145)
    $bgBrush = New-Object System.Drawing.Drawing2D.LinearGradientBrush($rect, $c1, $c2, [System.Drawing.Drawing2D.LinearGradientMode]::ForwardDiagonal)
    $g.FillEllipse($bgBrush, $rect)

    # Outer ring (cyan glow)
    $penGlow = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(210, 0, 215, 255), [float][Math]::Max(1.0, ($sz / 22.0)))
    $g.DrawEllipse($penGlow, $rect)

    # Stylized 3D Medical Iris / Eye
    $eyeX = [float]($sz * 0.18)
    $eyeY = [float]($sz * 0.28)
    $eyeW = [float]($sz * 0.64)
    $eyeH = [float]($sz * 0.44)
    $eyePen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(255, 240, 248, 255), [float][Math]::Max(1.2, ($sz / 18.0)))
    $g.DrawArc($eyePen, $eyeX, $eyeY, $eyeW, $eyeH, [float]0.0, [float]180.0)
    $g.DrawArc($eyePen, $eyeX, $eyeY, $eyeW, $eyeH, [float]180.0, [float]180.0)

    # Pupil / 3D Target Center
    $pupilSz = [float]($sz * 0.22)
    $pupilX = [float](($sz - $pupilSz) / 2.0)
    $pupilY = [float](($sz - $pupilSz) / 2.0)
    $pupilBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 0, 230, 180))
    $g.FillEllipse($pupilBrush, $pupilX, $pupilY, $pupilSz, $pupilSz)

    # 3D Coordinate / crosshair tick marks
    $chPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(230, 255, 80, 90), [float][Math]::Max(1.0, ($sz / 30.0)))
    $center = [float]($sz / 2.0)
    $offset = [float]($sz * 0.28)
    $g.DrawLine($chPen, $center, [float]($center - $offset), $center, [float]($center + $offset))
    $g.DrawLine($chPen, [float]($center - $offset), $center, [float]($center + $offset), $center)

    $penGlow.Dispose()
    $eyePen.Dispose()
    $pupilBrush.Dispose()
    $chPen.Dispose()
    $bgBrush.Dispose()
    $g.Dispose()

    $bitmaps += $bmp
}

# Combine into multi-layer ICO format stream
$ms = New-Object System.IO.MemoryStream
$bw = New-Object System.IO.BinaryWriter($ms)

# ICONHEADER
$bw.Write([UInt16]0) # Reserved
$bw.Write([UInt16]1) # Resource Type (1 for Icon)
$bw.Write([UInt16]$sizes.Count) # Number of images

$imgDataStreams = @()
$offset = 6 + ($sizes.Count * 16)

for ($i = 0; $i -lt $sizes.Count; $i++) {
    $bmp = $bitmaps[$i]
    $imgStream = New-Object System.IO.MemoryStream
    $bmp.Save($imgStream, [System.Drawing.Imaging.ImageFormat]::Png)
    $imgBytes = $imgStream.ToArray()
    $imgDataStreams += ,$imgBytes

    $szVal = if ($sizes[$i] -ge 256) { 0 } else { [byte]$sizes[$i] }
    
    # ICONDIRENTRY
    $bw.Write([byte]$szVal) # Width
    $bw.Write([byte]$szVal) # Height
    $bw.Write([byte]0)     # Color count
    $bw.Write([byte]0)     # Reserved
    $bw.Write([UInt16]1)   # Color planes
    $bw.Write([UInt16]32)  # Bits per pixel
    $bw.Write([UInt32]$imgBytes.Length) # Image data size in bytes
    $bw.Write([UInt32]$offset) # Offset to image data

    $offset += $imgBytes.Length
}

foreach ($bytes in $imgDataStreams) {
    $bw.Write($bytes)
}

$bw.Flush()
[System.IO.File]::WriteAllBytes($outputPath, $ms.ToArray())
$bw.Dispose()
$ms.Dispose()

foreach ($bmp in $bitmaps) {
    $bmp.Dispose()
}

Write-Host "Successfully generated MedEye3D icon: $outputPath"
