Add-Type @"
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

public class WindowFinder {
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    [DllImport("user32.dll")]
    public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);

    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);

    [DllImport("user32.dll")]
    public static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);

    [DllImport("user32.dll")]
    public static extern bool IsWindowVisible(IntPtr hWnd);

    public static List<string> FindWindowsForPID(uint targetPid) {
        var results = new List<string>();
        EnumWindows((hWnd, lParam) => {
            uint pid;
            GetWindowThreadProcessId(hWnd, out pid);
            if (pid == targetPid) {
                var sb = new StringBuilder(256);
                GetWindowText(hWnd, sb, 256);
                bool visible = IsWindowVisible(hWnd);
                results.Add("HWND=" + hWnd + " Visible=" + visible + " Title='" + sb.ToString() + "'");
            }
            return true;
        }, IntPtr.Zero);
        return results;
    }
}
"@

$p = Get-Process -Name "MedEye3D" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($p) {
    [WindowFinder]::FindWindowsForPID($p.Id)
} else {
    Write-Output "MedEye3D process not found"
}
