#!/usr/bin/env python3
"""
Comprehensive test script for HoloViews export in SSH environment
Tests multiple approaches to get static image export working
"""
import os
import sys
import subprocess
from pathlib import Path

# Try different display configurations
approaches = [
    {
        "name": "Using PyVirtualDisplay",
        "setup_func": "setup_pyvirtualdisplay",
        "description": "Uses pyvirtualdisplay to create a virtual X server"
    },
    {
        "name": "Direct environment variables",
        "setup_func": "setup_env_vars",
        "description": "Sets MOZ_HEADLESS and other environment variables"
    },
    {
        "name": "Using xvfb-run wrapper",
        "setup_func": None,
        "description": "Will use xvfb-run to execute the test",
        "run_with_xvfb": True
    }
]

def setup_pyvirtualdisplay():
    """Setup using PyVirtualDisplay"""
    print("Setting up PyVirtualDisplay...")
    try:
        from pyvirtualdisplay import Display
        display = Display(visible=0, size=(1920, 1080))
        display.start()
        print(f"✅ PyVirtualDisplay started: DISPLAY={os.environ.get('DISPLAY')}")
        return display
    except Exception as e:
        print(f"❌ PyVirtualDisplay failed: {e}")
        return None

def setup_env_vars():
    """Setup using environment variables only"""
    print("Setting up environment variables...")
    os.environ['MOZ_HEADLESS'] = '1'
    os.environ['DISPLAY'] = ':99'
    os.environ['XVFB_WHD'] = '1920x1080x24'
    print("✅ Environment variables set:")
    print(f"   MOZ_HEADLESS: {os.environ.get('MOZ_HEADLESS')}")
    print(f"   DISPLAY: {os.environ.get('DISPLAY')}")
    return True

def test_holoviews_export(approach_name, setup_result=None):
    """Test HoloViews export with current setup"""
    print(f"\n--- Testing HoloViews export with {approach_name} ---")

    try:
        import pandas as pd
        import numpy as np
        import holoviews as hv

        # Enable Bokeh backend
        hv.extension("bokeh")

        # Create simple test data
        np.random.seed(42)
        data = pd.DataFrame({
            'x': np.random.randn(50),
            'y': np.random.randn(50),
            'category': np.random.choice(['A', 'B'], 50)
        })

        # Create a simple scatter plot
        scatter = hv.Scatter(data, kdims=['x'], vdims=['y', 'category']).opts(
            color='category',
            width=400,
            height=400,
            title=f"Test Plot - {approach_name}"
        )

        # Test HTML export (should always work)
        html_file = f"test_{approach_name.lower().replace(' ', '_')}.html"
        print(f"Saving HTML: {html_file}")
        hv.save(scatter, html_file)
        print("✅ HTML export successful")

        # Test PNG export
        png_file = f"test_{approach_name.lower().replace(' ', '_')}.png"
        print(f"Attempting PNG export: {png_file}")

        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("PNG export timed out")

        # Set timeout for PNG export
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout

        try:
            hv.save(scatter, png_file)
            signal.alarm(0)  # Cancel timeout
            print("✅ PNG export successful!")

            # Check if file exists and has content
            if Path(png_file).exists() and Path(png_file).stat().st_size > 0:
                print(f"   File size: {Path(png_file).stat().st_size} bytes")
                return True
            else:
                print("❌ PNG file is empty or doesn't exist")
                return False

        except TimeoutError:
            signal.alarm(0)
            print("❌ PNG export timed out (30s)")
            return False
        except Exception as e:
            signal.alarm(0)
            print(f"❌ PNG export failed: {e}")
            return False

    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        return False
    finally:
        # Cleanup
        if setup_result and hasattr(setup_result, 'stop'):
            print("Stopping virtual display...")
            setup_result.stop()

def test_selenium_directly():
    """Test selenium directly to see what's happening"""
    print("\n--- Testing Selenium directly ---")

    try:
        from selenium import webdriver
        from selenium.webdriver.firefox.options import Options

        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')

        print("Creating Firefox webdriver...")
        driver = webdriver.Firefox(options=options)
        print("✅ Firefox webdriver created successfully")

        print("Testing basic navigation...")
        driver.get("data:text/html,<html><body><h1>Test</h1></body></html>")
        print("✅ Navigation successful")

        print("Taking screenshot...")
        driver.save_screenshot("selenium_test.png")
        print("✅ Screenshot successful")

        driver.quit()
        print("✅ Selenium test completed successfully")
        return True

    except Exception as e:
        print(f"❌ Selenium test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=== HoloViews Export Test Suite ===")
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"DISPLAY: {os.environ.get('DISPLAY', 'Not set')}")

    # Test selenium first
    selenium_works = test_selenium_directly()

    if not selenium_works:
        print("\n⚠️  Selenium doesn't work directly, HoloViews export likely won't work either")
        print("Let's try the xvfb-run approach...")

        # Test with xvfb-run
        print("\n--- Testing with xvfb-run ---")
        try:
            result = subprocess.run([
                'xvfb-run', '-a', '--server-args=-screen 0 1920x1080x24',
                'python', '-c', '''
import pandas as pd
import numpy as np
import holoviews as hv
import os
print("Running inside xvfb-run...")
print(f"DISPLAY: {os.environ.get('DISPLAY')}")
hv.extension("bokeh")
data = pd.DataFrame({"x": [1,2,3], "y": [1,4,2]})
scatter = hv.Scatter(data, kdims=["x"], vdims=["y"])
hv.save(scatter, "xvfb_test.png")
print("✅ xvfb-run PNG export successful!")
'''
            ], capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                print("✅ xvfb-run approach works!")
                print("Output:", result.stdout)

                # Check if file was created
                if Path("xvfb_test.png").exists():
                    print(f"✅ PNG file created: {Path('xvfb_test.png').stat().st_size} bytes")

                    # Provide solution
                    print("\n🎉 SOLUTION FOUND!")
                    print("To export HoloViews plots as PNG/SVG in SSH environments:")
                    print("1. Use xvfb-run to wrap your Python script:")
                    print("   xvfb-run -a python your_script.py")
                    print("2. Or modify your save function to use xvfb-run internally")

                    return True

            else:
                print("❌ xvfb-run approach failed")
                print("Error:", result.stderr)

        except subprocess.TimeoutExpired:
            print("❌ xvfb-run approach timed out")
        except Exception as e:
            print(f"❌ xvfb-run approach failed: {e}")

    else:
        print("\n🎉 Selenium works! Let's test HoloViews approaches...")

        success_count = 0
        for approach in approaches:
            if approach.get("run_with_xvfb"):
                continue  # Skip xvfb test since selenium already works

            setup_func = globals().get(approach["setup_func"])
            if setup_func:
                setup_result = setup_func()
                success = test_holoviews_export(approach["name"], setup_result)
                if success:
                    success_count += 1
            else:
                print(f"Setup function {approach['setup_func']} not found")

        if success_count > 0:
            print(f"\n🎉 SUCCESS! {success_count} approach(es) worked!")
        else:
            print("\n❌ No approaches worked, falling back to xvfb-run solution")

    print("\n=== Test Complete ===")
    return False

if __name__ == "__main__":
    main()
