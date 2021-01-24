SHELL = bash  # To make popd available

## docs/api/build/html/index.html : Documentation for the api
docs/api/build/html/index.html : docs/api/source/conf.py apidoc_source
	pushd docs/api; rm -rf build; make html; popd
	@echo To view: firefox --no-remote docs/api/build/html/index.html

## test                           : Discover and run all tests in hmm
.PHONY : test
test :
	pytest hmm

## check-types
.PHONY : check-types
check-types:
	mypy hmm/


## coverage                       : make test coverage report in htmlcov/
.PHONY : coverage
coverage :
	rm -rf .coverage htmlcov
	coverage run -m pytest hmm/tests/
	coverage html  --omit=/nix/store*


## yapf                           : Force google format on all python code
.PHONY : yapf
yapf :
	yapf -i --recursive --style "google" hmm

## test_standards                 : Build api documentation and run pylint
.PHONY : test_standards
test_standards :
	pytest test_coding_standards.py

## pylintrc                       : Fetch Google's rules for python style
pylintrc:
	wget --no-check-certificate https://google.github.io/styleguide/pylintrc

## clean                          : Remove machine generated files
.PHONY : clean
clean :
	rm -f *.npy

## help                           : Print comments on targets from makefile
.PHONY : help
help : Makefile
	@sed -n 's/^## / /p' $<

###---------------
### Local Variables:
### mode: makefile
### End:
