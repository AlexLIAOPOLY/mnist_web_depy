#!/usr/bin/env python
"""
MNIST Web Application Launcher
A simple script to run the MNIST web application with customizable options.
"""

import argparse
import os
import sys
from pathlib import Path

def main():
    """Parse command line arguments and start the MNIST web application."""
    parser = argparse.ArgumentParser(description='Run the MNIST Web Application')
    parser.add_argument('--host', default='127.0.0.1', 
                        help='The host to bind the server to')
    parser.add_argument('--port', type=int, default=5000, 
                        help='The port to bind the server to')
    parser.add_argument('--debug', action='store_true', 
                        help='Run the application in debug mode')
    parser.add_argument('--preload', action='store_true',
                        help='Preload the MNIST dataset')
    
    args = parser.parse_args()
    
    # Add the project root to Python path
    project_root = Path(__file__).parent.absolute()
    sys.path.append(str(project_root))
    
    # Set environment variable to indicate preloading
    if args.preload:
        os.environ['MNIST_PRELOAD_DATA'] = '1'
    
    # Import the Flask app
    from mnist_web.app import app
    
    # Run the application
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main() 