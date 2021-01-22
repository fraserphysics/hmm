'''Steps for running the tests defined here:

from this directory issue "python test_coding_standards.py" or
"py.test --pdb test_coding_standards.py"

'''
# pylint: disable = invalid-name, missing-docstring
import subprocess
import unittest
import os.path
import pdb

import hmm


class BaseClass(unittest.TestCase):
    root = os.path.dirname(hmm.__path__[0])


class Test_lint(BaseClass):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_pylint(self):
        '''Test that projects/modules specified conform to coding standards
        specified in rc_file.  Alternatively:
        "pylint --rcfile=pylintrc hmm"

        '''
        rcfile_path = os.path.join(self.root, 'pylintrc')
        args = [
            'pylint', '--rcfile={0}'.format(rcfile_path),
            os.path.join(self.root, 'hmm')
        ]
        with subprocess.Popen(args,
                              cwd=self.root,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as pipe:
            pipe.wait()
            # self.assertEqual(pipe.stdout.read(), '') TODO ?
            self.assertEqual(pipe.returncode, 0)


class Test_doc(BaseClass):  # TODO: Change Foo to Test
    def test_api_doc(self):
        '''Test that docstrings are OK.  Alternatively: "cd docs/api; make
        html"

        '''

        args = 'make apidoc_source'.split()
        with subprocess.Popen(args,
                              cwd=self.root,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as pipe:
            pipe.wait()
            self.assertEqual(pipe.returncode, 0)
        with subprocess.Popen('make html'.split(),
                            cwd=os.path.join(self.root, 'docs', 'api'),
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE) as pipe:
            pipe.wait()
            # self.assertEqual(pipe.stderr.read(), '') TODO ?
            self.assertEqual(pipe.returncode, 0)


if __name__ == '__main__':
    unittest.main()
