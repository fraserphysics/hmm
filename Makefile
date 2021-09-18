SHELL = bash  # To make popd available

help:

## docs                           : Make all the docmentation
.PHONY: docs
docs: docs/api/build/html/index.html docs/manual/build/html/index.html

## doc_server                     : Start doc server on https://localhost:8000
.PHONY: doc_server
doc_server:
	sphinx-autobuild --ignore docs/manual/build/ docs/manual/source/ docs/manual/build/html

## doc_server                     : Start API doc server on https://localhost:8000
.PHONY: doc_server_api
doc_server_api:
	sphinx-autobuild --ignore docs/api/build/ docs/api/source/ docs/api/build/html

## docs/api/build/html/index.html : Documentation for the api
docs/api/build/html/index.html : docs/api/source/conf.py
	cd docs/api; rm -rf build; make html
	@echo To view: firefox --no-remote docs/api/build/html/index.html

## docs/manual/build/html/index.html : User documentation
docs/manual/build/html/index.html : docs/api/source/conf.py
	cd docs/manual; rm -rf build; make html
	@echo To view: firefox --no-remote docs/manual/build/html/index.html

## LICENSE.txt                    : gpl-3.0 License for distributing the hmm software
LICENSE.txt:
	wget https://www.gnu.org/licenses/gpl-3.0.txt -O $@

## test                           : Discover and run all tests in hmm
.PHONY : test, test_all
test :
	pytest

## test_all                       : Test against all supported versions of python
test_all:
	nox

## check-types                    : Checks type hints
.PHONY : check-types
check-types:
	mypy --no-strict-optional hmm/ tests/
# --no-strict-optional allows None as default value

## coverage                       : make test coverage report in htmlcov/
.PHONY : coverage
coverage :
	rm -rf .coverage htmlcov
	coverage run -m pytest tests
	coverage html  --omit=/nix/store*

## yapf                           : Force google format on all python code
.PHONY : yapf
yapf :
	yapf -i --recursive --style "google" hmm

## test_standards                 : Build api documentation and run pylint
.PHONY : test_standards
test_standards :
	pytest test_coding_standards.py
# Debug with: "pylint --rcfile=pylintrc hmm tests" and "cd docs/api; make html"

## pylintrc                       : Fetch Google's rules for python style
pylintrc:
	git -c http.sslVerify=false clone https://chromium.googlesource.com/chromium/tools/depot_tools.git
	cp depot_tools/pylintrc .
	echo Edit: max-line-length=86, no pylint_quotes


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
