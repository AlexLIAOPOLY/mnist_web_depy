#!/usr/bin/env python
"""
Production runner for the MNIST Web Application
This script is used to run the application in production environments
like Render and Heroku.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

# Set environment variable to preload data
os.environ['MNIST_PRELOAD_DATA'] = '1'

# Import the Flask app
from mnist_web.app import app

if __name__ == '__main__':
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get('PORT', 5000))
    
    # Run the application
    app.run(host='0.0.0.0', port=port) 