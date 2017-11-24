all: compile
full: compile install clean

ifndef INSTALL_DIR
INSTALL_DIR=./built-lib/bayesfilter/
endif

ifndef SWIG
SWIG=swig
endif

ifndef DOXYGEN
DOXYGEN=doxygen
endif

DOC_DIR=../doc/
SRC_DIR = ../src/

compile: bayesfilter.i setup.py $(SRC_DIR)*
	@echo "\ngenerating doc\n"
	@(cd ../doc/ && $(DOXYGEN))
	@echo "\nCompiling Library\n"
	python doxy2swig.py $(DOC_DIR)xml/index.xml mhmm_doc.i
	$(SWIG) -c++ $(SWIGOPT) -I$(SRC_DIR) -python bayesfilter.i
	python setup.py build_ext --inplace

install:
	mkdir -p $(INSTALL_DIR)bayesfilter/
	mv bayesfilter.py $(INSTALL_DIR)bayesfilter/
	mv _bayesfilter.so $(INSTALL_DIR)bayesfilter/
	cp __init__.py $(INSTALL_DIR)bayesfilter/

clean:
	rm -f *.cxx *.so bayesfilter.py
	rm -rf build/