from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy

numpy_inc = numpy.get_include()

extensions = [
	Extension("hidden_markov.markov_chain",
			["hidden_markov/markov_chain.pyx"],
		include_dirs = [numpy_inc],
		),
	Extension("hidden_markov.markov_random_field",
			["hidden_markov/markov_random_field.pyx"],
		include_dirs = [numpy_inc],
		),
	Extension("hidden_markov.likelihood",
			["hidden_markov/likelihood.pyx"],
		include_dirs = [numpy_inc],
		),
	Extension("hidden_markov.ising",
			["hidden_markov/ising.pyx"],
		extra_compile_args=['-fopenmp'],
		extra_link_args=['-fopenmp'],
		include_dirs = [numpy_inc],
		),
	Extension("mrf.mrf_parallel_icm", ["mrf/mrf_parallel_icm.pyx"],
		include_dirs = [numpy_inc],
		extra_compile_args=['-fopenmp'],
		extra_link_args=['-fopenmp'],
		),
	Extension("mrf.mrf_parallel_icm_3d", ["mrf/mrf_parallel_icm_3d.pyx"],
		include_dirs = [numpy_inc],
		extra_compile_args=['-fopenmp'],
		extra_link_args=['-fopenmp'],
		),
	Extension("mrf.mrf_hierarchical_1d", ["mrf/mrf_hierarchical_1d.pyx"],
		include_dirs = [numpy_inc],
#		extra_compile_args=['-fopenmp'],
#		extra_link_args=['-fopenmp'],
		),
	]


setup(
	name = "Markov chains and random fields",
	ext_modules = cythonize(extensions),
	packages=['hidden_markov','mrf'],
	zip_safe=False,
)

#__init__.py  markov_chain.pyx  sequence_segmenter.py
