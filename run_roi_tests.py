#!/usr/bin/env python3
"""
Test runner for ROI management functionality.
Runs all ROI-related tests and provides a summary.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_tests():
    """Run all ROI-related tests."""
    print("Running ROI Management Tests...")
    print("=" * 50)
    
    # Add src to Python path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    # Test files to run
    test_files = [
        "tests/test_roi_manager.py",
        "tests/test_cv_service_roi.py"
    ]
    
    total_passed = 0
    total_failed = 0
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"❌ Test file not found: {test_file}")
            continue
        
        print(f"\n🧪 Running {test_file}...")
        print("-" * 30)
        
        try:
            # Run pytest on the specific file
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                test_file, 
                "-v", 
                "--tb=short",
                "--no-header"
            ], capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                print(f"✅ {test_file} - All tests passed")
                # Count passed tests from output
                passed = result.stdout.count(" PASSED")
                total_passed += passed
                print(f"   Passed: {passed}")
            else:
                print(f"❌ {test_file} - Some tests failed")
                # Count failed tests from output
                failed = result.stdout.count(" FAILED")
                passed = result.stdout.count(" PASSED")
                total_failed += failed
                total_passed += passed
                print(f"   Passed: {passed}, Failed: {failed}")
                
                # Show error details
                if result.stdout:
                    print("STDOUT:")
                    print(result.stdout)
                if result.stderr:
                    print("STDERR:")
                    print(result.stderr)
        
        except Exception as e:
            print(f"❌ Error running {test_file}: {e}")
            total_failed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"✅ Total Passed: {total_passed}")
    print(f"❌ Total Failed: {total_failed}")
    
    if total_failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return False

def check_dependencies():
    """Check if required dependencies are available."""
    required_packages = ["pytest", "numpy", "opencv-python"]
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required packages:")
        for package in missing:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True

if __name__ == "__main__":
    print("ROI Management Test Suite")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Run tests
    success = run_tests()
    
    sys.exit(0 if success else 1)