SHELL = bash  # To make popd available

## docs/api/build/html/index.html : Documentation for the api
docs/api/build/html/index.html : docs/api/source/conf.py
	pushd docs/api; rm -rf build; make html; popd
	@echo To view: firefox --no-remote docs/api/build/html/index.html

## gpl.txt                        : License for distributing the hmm software
gpl.txt:
	wget https://www.gnu.org/licenses/gpl-3.0.txt -O $@

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
