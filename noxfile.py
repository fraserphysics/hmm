import os

import nox

# Do not use anything installed in the site local directory (~/.local for example) which
# might have been installed by pip install --user.  These can prevent the install here
# from pulling in the correct packages, thereby mucking up tests later on.
# See https://stackoverflow.com/a/51640558/1088938
os.environ["PYTHONNOUSERSITE"] = "1"

args = dict(python=["3.7", "3.8", "3.9"], reuse_venv=True)


@nox.session(**args)
def test(session):
    session.install(".[test]")
    session.run("pytest")
