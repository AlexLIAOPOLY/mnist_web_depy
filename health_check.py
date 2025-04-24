#!/usr/bin/env python
"""
Health check script for the MNIST Web Application
This script verifies that all required dependencies are installed
and the application can start correctly.
"""

import importlib
import os
import sys
from pathlib import Path

def check_dependency(package_name):
    """Check if a Python package is installed."""
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name} installed")
        return True
    except ImportError:
        print(f"❌ {package_name} not installed")
        return False

def check_file_exists(file_path):
    """Check if a file exists."""
    if os.path.exists(file_path):
        print(f"✅ {file_path} exists")
        return True
    else:
        print(f"❌ {file_path} missing")
        return False

def main():
    """Run health checks for the MNIST Web Application."""
    print("Running MNIST Web Application health checks...")
    print("\n1. Checking Python version...")
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} (recommended: 3.8+)")
    else:
        print(f"⚠️ Python {python_version.major}.{python_version.minor}.{python_version.micro} (recommended: 3.8+)")
    
    print("\n2. Checking required packages...")
    required_packages = [
        'flask', 'numpy', 'pillow', 'scikit-learn', 'scipy', 
        'matplotlib', 'requests'
    ]
    
    all_packages_installed = True
    for package in required_packages:
        if not check_dependency(package):
            all_packages_installed = False
    
    print("\n3. Checking required files...")
    project_root = Path(__file__).parent.absolute()
    required_files = [
        os.path.join(project_root, 'run_mnist_web.py'),
        os.path.join(project_root, 'mnist_web', 'app.py'),
        os.path.join(project_root, 'requirements.txt'),
    ]
    
    all_files_exist = True
    for file_path in required_files:
        if not check_file_exists(file_path):
            all_files_exist = False
    
    print("\n4. Checking application startup...")
    try:
        sys.path.append(str(project_root))
        from mnist_web.app import app
        print("✅ Application imports successfully")
        
        # Test creating a client
        client = app.test_client()
        response = client.get('/')
        if response.status_code == 200:
            print("✅ Application responds to requests")
        else:
            print(f"⚠️ Application returned status code {response.status_code} for root URL")
    except Exception as e:
        print(f"❌ Error starting application: {str(e)}")
        all_files_exist = False
    
    print("\n----- Summary -----")
    if all_packages_installed and all_files_exist:
        print("✅ All checks passed! The application should work correctly.")
        return 0
    else:
        print("⚠️ Some checks failed. Please resolve the issues above before deploying.")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 