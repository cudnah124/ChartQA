"""
Run all tests
"""

import subprocess
import sys
import os

def run_test(test_file):
    """Run a single test file"""
    print(f"\n{'=' * 60}")
    print(f"Running: {test_file}")
    print('=' * 60)
    
    result = subprocess.run(
        [sys.executable, test_file],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=False
    )
    
    return result.returncode == 0

def main():
    """Run all tests"""
    print("\n🧪 Running All Tests\n")
    
    tests = [
        "tests/test_config.py",
        "tests/test_data_loader.py",
        "tests/test_collator.py",
        "tests/test_format.py"
    ]
    
    results = {}
    for test in tests:
        test_name = os.path.basename(test)
        results[test_name] = run_test(test)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed!\n")
        return 0
    else:
        print("\n❌ Some tests failed!\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
