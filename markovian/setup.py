from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy

numpy_inc = numpy.get_include()

extensions = [
    Extension("hidden_markov.markov_chain",
			["hidden_markov/markov_chain.pyx"],
#			"hidden_markov/markov_random_field.pyx"],
        include_dirs = [numpy_inc],
		),
    Extension("hidden_markov.markov_random_field",
#		["hidden_markov/markov_chain.pyx",
			["hidden_markov/markov_random_field.pyx"],
        include_dirs = [numpy_inc],
		)
	]


setup(
    name = "Markov chains and random fields",
    ext_modules = cythonize(extensions),
	packages=['hidden_markov'],
)

#__init__.py  markov_chain.pyx  sequence_segmenter.py
