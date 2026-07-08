"""
Power BI Auto-Refresh Script
=============================
Automatically refreshes Power BI Desktop dashboard every N minutes
No manual refresh needed!

Requirements:
- Power BI Desktop installed
- Dashboard file (.pbix) exists
- Power BI Desktop running

Usage:
    python powerbi_auto_refresh.py
"""

import os
import time
import subprocess
from datetime import datetime
import sys

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION - EDIT THESE
# ═══════════════════════════════════════════════════════════════════════════

PBIX_FILE = r"E:\Akshat E&TC\FY\CP\dashboard.pbix"
REFRESH_INTERVAL_MINUTES = 5  # How often to refresh
POWER_BI_PATH = r"C:\Program Files\Microsoft Power BI Desktop\bin\PBIDesktop.exe"

# ═══════════════════════════════════════════════════════════════════════════
# AUTO-REFRESH FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def refresh_powerbi():
    """
    Refresh Power BI Desktop file using keyboard shortcut
    Sends Ctrl+R to Power BI Desktop window
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] 🔄 Triggering Power BI refresh...")
    
    # PowerShell script to send Ctrl+R to Power BI Desktop
    ps_script = """
    Add-Type -AssemblyName System.Windows.Forms
    $wshell = New-Object -ComObject wscript.shell
    
    # Find Power BI window
    $powerbi = Get-Process -Name "PBIDesktop" -ErrorAction SilentlyContinue
    if ($powerbi) {
        # Bring Power BI to foreground (briefly)
        $wshell.AppActivate("Power BI Desktop")
        Start-Sleep -Milliseconds 500
        
        # Send Ctrl+R (Refresh shortcut in Power BI)
        $wshell.SendKeys("^r")
        Write-Host "✓ Refresh triggered"
    } else {
        Write-Host "✗ Power BI Desktop not running"
        exit 1
    }
    """
    
    try:
        result = subprocess.run(
            ["powershell", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if "Refresh triggered" in result.stdout:
            print("  ✓ Refresh command sent successfully")
            return True
        else:
            print("  ✗ Power BI Desktop not running")
            return False
            
    except subprocess.TimeoutExpired:
        print("  ✗ Timeout - PowerShell script took too long")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def ensure_powerbi_running():
    """
    Check if Power BI Desktop is running
    If not, try to open it with the dashboard file
    """
    print("📊 Checking Power BI Desktop status...")
    
    ps_check = "Get-Process -Name 'PBIDesktop' -ErrorAction SilentlyContinue"
    result = subprocess.run(
        ["powershell", "-Command", ps_check],
        capture_output=True,
        text=True
    )
    
    if not result.stdout.strip():
        print("  ⚠ Power BI Desktop not running")
        
        if os.path.exists(POWER_BI_PATH) and os.path.exists(PBIX_FILE):
            print(f"  → Opening Power BI Desktop with dashboard...")
            try:
                subprocess.Popen([POWER_BI_PATH, PBIX_FILE])
                print("  → Waiting 30 seconds for Power BI to start...")
                time.sleep(30)
                return True
            except Exception as e:
                print(f"  ✗ Failed to open Power BI: {e}")
                return False
        else:
            print(f"  ✗ Power BI or dashboard file not found")
            print(f"     Power BI path: {POWER_BI_PATH}")
            print(f"     Dashboard path: {PBIX_FILE}")
            return False
    else:
        print("  ✓ Power BI Desktop is running")
        return True


def verify_configuration():
    """Verify all required files exist before starting"""
    print("🔍 Verifying configuration...")
    
    errors = []
    
    # Check dashboard file
    if not os.path.exists(PBIX_FILE):
        errors.append(f"Dashboard file not found: {PBIX_FILE}")
    else:
        size_mb = os.path.getsize(PBIX_FILE) / (1024 * 1024)
        print(f"  ✓ Dashboard file found ({size_mb:.1f} MB)")
    
    # Check Power BI Desktop
    if not os.path.exists(POWER_BI_PATH):
        errors.append(f"Power BI Desktop not found: {POWER_BI_PATH}")
        errors.append("  Install from: https://powerbi.microsoft.com/desktop/")
    else:
        print(f"  ✓ Power BI Desktop found")
    
    # Check PowerShell
    try:
        subprocess.run(
            ["powershell", "-Command", "Write-Host 'OK'"],
            capture_output=True,
            timeout=5
        )
        print(f"  ✓ PowerShell available")
    except:
        errors.append("PowerShell not available")
    
    if errors:
        print("\n❌ Configuration errors:")
        for error in errors:
            print(f"   {error}")
        return False
    
    print("  ✓ All checks passed\n")
    return True


def print_header():
    """Print welcome header"""
    print("=" * 70)
    print(" " * 5 + "Power BI Auto-Refresh Service")
    print("=" * 70)
    print(f"Dashboard:        {os.path.basename(PBIX_FILE)}")
    print(f"Refresh Interval: {REFRESH_INTERVAL_MINUTES} minutes")
    print(f"Start Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()


def main():
    """Main auto-refresh loop"""
    print_header()
    
    # Verify configuration
    if not verify_configuration():
        print("\n❌ Cannot start auto-refresh service")
        print("   Please fix configuration errors above")
        sys.exit(1)
    
    # Ensure Power BI is running
    if not ensure_powerbi_running():
        print("\n❌ Cannot start auto-refresh - Power BI Desktop not running")
        print("   Please open Power BI Desktop and try again")
        sys.exit(1)
    
    print()
    print("=" * 70)
    print("🚀 Auto-refresh service started")
    print("   Press Ctrl+C to stop")
    print("=" * 70)
    print()
    
    refresh_count = 0
    
    try:
        while True:
            # Perform refresh
            success = refresh_powerbi()
            
            if success:
                refresh_count += 1
                next_refresh = datetime.now().timestamp() + (REFRESH_INTERVAL_MINUTES * 60)
                next_refresh_time = datetime.fromtimestamp(next_refresh).strftime('%H:%M:%S')
                
                print(f"  📊 Total refreshes: {refresh_count}")
                print(f"  ⏰ Next refresh at: {next_refresh_time}")
                print()
            else:
                print("  ⚠ Refresh failed - will retry in 1 minute")
                time.sleep(60)
                continue
            
            # Wait until next refresh
            print(f"  💤 Sleeping for {REFRESH_INTERVAL_MINUTES} minutes...")
            print()
            time.sleep(REFRESH_INTERVAL_MINUTES * 60)
            
    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("🛑 Auto-refresh service stopped by user")
        print(f"   Total refreshes performed: {refresh_count}")
        print(f"   Runtime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)


if __name__ == "__main__":
    main()
