#!/usr/bin/env python
"""
Basic tests for the MNIST web application
"""
import unittest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from mnist_web.app import app

class TestMnistWebApp(unittest.TestCase):
    """Test cases for the MNIST web application."""

    def setUp(self):
        """Set up test client."""
        self.app = app.test_client()
        self.app.testing = True

    def test_home_page(self):
        """Test the home page."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'MNIST Web Application', response.data)

    def test_train_page(self):
        """Test the train page."""
        response = self.app.get('/train')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Train Model', response.data)

    def test_draw_page(self):
        """Test the draw page."""
        response = self.app.get('/draw')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Test Predictions', response.data)

    def test_explore_page(self):
        """Test the explore page."""
        response = self.app.get('/explore')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Explore Dataset', response.data)

    def test_models_page(self):
        """Test the models page."""
        response = self.app.get('/models')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Compare Model Performance', response.data)

if __name__ == '__main__':
    unittest.main() 