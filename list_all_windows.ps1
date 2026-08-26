Add-Type @"
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

public class WinLister {
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    [DllImport("user32.dll")]
    public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);

    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);

    [DllImport("user32.dll")]
    public static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);

    [DllImport("user32.dll")]
    public static extern int GetClassName(IntPtr hWnd, StringBuilder lpString, int nMaxCount);

    [DllImport("user32.dll")]
    public static extern bool IsWindowVisible(IntPtr hWnd);

    public static List<string> GetAllVisibleWindows() {
        var results = new List<string>();
        EnumWindows((hWnd, lParam) => {
            uint pid;
            GetWindowThreadProcessId(hWnd, out pid);
            var sbTitle = new StringBuilder(256);
            GetWindowText(hWnd, sbTitle, 256);
            var sbClass = new StringBuilder(256);
            GetClassName(hWnd, sbClass, 256);
            bool visible = IsWindowVisible(hWnd);
            if (sbTitle.Length > 0 || sbClass.ToString().Contains("GLFW")) {
                results.Add("PID=" + pid + " HWND=" + hWnd + " Visible=" + visible + " Class='" + sbClass.ToString() + "' Title='" + sbTitle.ToString() + "'");
            }
            return true;
        }, IntPtr.Zero);
        return results;
    }
}
"@

[WinLister]::GetAllVisibleWindows() | ForEach-Object { Write-Output $_ }
