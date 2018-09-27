"""Run PEP8 on all Python files in this directory and
subdirectories as part of the tests."""
import os
import unittest
import pycodestyle


class TestPep8(unittest.TestCase):
    """Run PEP8 on all project files."""
    def test_pep8(self):
        style = pycodestyle.StyleGuide(quiet=False,
                                       config_file='tox.ini')
        for root, dirs, files in os.walk('dlopt'):
            python_files = [os.path.join(root, f)
                            for f in files if f.endswith('.py')]
            style.check_files(python_files)
        n = style.check_files().total_errors
        self.assertEqual(n, 0, 'PEP8 style errors: %d' % n)
