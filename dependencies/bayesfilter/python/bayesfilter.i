%module(docstring="Non-linear Bayesian Filtering for EMG envelope extraction") bayesfilter

%{
	#define SWIG_FILE_WITH_INIT
	#include "BayesianFilter.h"
%}

%exception {
    try {
        $action
    }
    catch (exception const& e) {
        PyErr_SetString(PyExc_IndexError,e.what());
        SWIG_fail;
    }
}

%include numpy.i
%include "BayesianFilter.h"

%init %{
	import_array();
%}

%apply (int DIM1, double* IN_ARRAY1) { (int _inchannels, double *observation) };
%apply (int DIM1, double* ARGOUT_ARRAY1) { (int _outchannels, double *_output) };