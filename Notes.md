Suggestions
===========

# Organization

It is generally advisable to place your code in a `src/` directory so it is only used
and tested if it is properly installed.

* https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure
* https://setuptools.readthedocs.io/en/latest/userguide/declarative_config.html?highlight=src#using-a-src-layout
* https://setuptools.readthedocs.io/en/latest/userguide/declarative_config.html?highlight=src#using-a-src-layout
It seems like your documentation expects the tests to be part of the package, so I have
moved them to `src/hmm/tests` as discussed here:

* https://docs.pytest.org/en/latest/explanation/goodpractices.html?highlight=src#tests-as-part-of-application-code

The tests would be run with:

```bash
nox
```

# Testing

In order to test against multiple versions of python, I find [Nox] very useful.  This is
configured in `noxfile.py`, which specifies which version of python to test against and
how to run the tests.  The way this is currently implemented requires that you have
executable for `python3.7`, `python3.8`, and `python3.9` available in your build/test
environment.  An alternative is to use [Conda] to pull these in as needed.  Let me know if
you would like to do the latter.

To test against all version of python, simply run:

```bash
nox
```

<details><summary>What I do.</summary>

What I actually do on my system is to use [Conda] to make a set of basic environments
with just python, and then I link these to my `~/.local/bin/` folder:

```bash
for py in 3.7 3.8 3.9; do
  _env=py${py}
  conda create -n ${_env} python=$py
  _env_path="$(conda info -e | grep ^${_env} | while read _name _path; do echo ${_path}; done)"
  _python="${_env_path}/bin/python${py}"
  ln -s "${_python}" ~/.local/bin/
done
```

</details>

## PyTests

A couple of simplifications can be made:

1. There is no need to use unitest: see
   https://docs.pytest.org/en/latest/how-to/xunit_setup.html. Some changes are needed:
   the minimal is to rename `setUp()` to `setup_class()` and make this a
   `@classmethod`.  (Also requires making those methods used here `@classmethods` like
   `make_y_mods()`.)  The preferred approach is to use
   [fixtures](https://docs.pytest.org/en/latest/reference/fixtures.html#fixture) but it
   is a bit complicated.
2. Use plain `assert`: this will allow `pytest` to provide more useful information about
   failures.

# Documentation

Do you really want two separate documents for the manual and API?  I would probably
merge these into a single document -- neither is that big.  Currently I am just putting
the manual on [RTD].

## Read the Docs

I added a file `.readthedocs.yaml` which specifies how to build to documentation on
[RTD].  You will need to create an account on [RTD] and a project that points to your
[GitHub] or [GitLab] repo.  When you push, the documentation should be updated.

# Build

I would provide some extras:

* `[test]`: Include `pytest` dependencies so that users can use `pip install .[test]`
  and then run the tests.
* `[doc]`: Include whatever is needed to build the docs.

To facilitate working with the different environments, I still might consider using
[Poetry].  For example, to build the docs right now, one
first has to create an environment, then `pip install .[doc]`, then build.  With
[Poetry] this could be done in a single command.

# CI

CI still needs to be setup.  Let me know if you want to use [GitHub] or [GitLab].
   
# Makefile

* I think there is no need to use `popd`.  Make does not remember directory changes
  between lines.
* `make clean` should clean up docs too.  (I originally built the docs, but since I had
  not installed `hmm`, this failed.  However, now it is difficult to rebuild once I have
      it installed.)

[Nox]: <https://nox.thea.codes> "nox is a command-line tool that automates testing in multiple Python environments, similar to tox. Unlike tox, Nox uses a standard Python file for configuration."
[RTD]: <https://readthedocs.org> "Read the Docs"
[Poetry]: <https://python-poetry.org>
