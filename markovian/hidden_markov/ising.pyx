# Need to document

import numpy 
cimport numpy
cimport cython
from cython.parallel import prange

from libc.math cimport exp, sqrt
#from libc.math cimport M_PI

import numpy

# This function is in place for error checking 
def ising_iter_gibbs(mrf,img,beta,ewt,cwt,lwt,hwt):
	ising_iter_gibbs_c(mrf,img,beta,ewt,cwt,lwt,hwt)

	return


@cython.boundscheck(False)
@cython.cdivision(True)
cdef ising_iter_gibbs_c(
		numpy.ndarray[numpy.int64_t,ndim=2] mrf,
		numpy.ndarray[numpy.int64_t,ndim=2] img,
		double beta, double ewt, double cwt, double lwt, double hwt):
	# this implementation uses (0,1) values to code states rather than
	# (-1,1) values in literature
	# the underlying mrf and observed values are assumed to be binary
	# beta is analogous to temperature
	# ewt,cwt,lwt are effectively for computation purposes weights for
	# edges, corners and observed aka likelihood
	# hwt is the proportion expected to code as zero

	cdef int N = mrf.shape[0]
	cdef int P = mrf.shape[1]
	cdef int i,j
	cdef double sum_e, sum_c, sum_ec
	cdef double prob

	cdef double[:,:] urand = numpy.random.uniform(size=(N,P))

#	cdef numpy.ndarray[numpy.float64_t,ndim=2] urand = \
#				numpy.random.uniform(size=(N,P))

	# corners
	sum_ec = ewt*(mrf[0,1] + mrf[1,0]) + cwt*mrf[1,1]
	sum_ec += lwt*img[0,0] - hwt
	prob = 1.0/(1.0 + exp(-beta*sum_ec))
	if prob < urand[0,0]: mrf[0,0] = 0
	else: mrf[0,0] = 1
	
	sum_ec = ewt*(mrf[0,P-2] + mrf[1,P-1]) + cwt*mrf[1,P-2]
	sum_ec += lwt*img[0,P-1] - hwt
	prob = 1.0/(1.0 + exp(-beta*sum_ec))
	if prob < urand[0,P-1]: mrf[0,P-1] = 0
	else: mrf[0,P-1] = 1
	
	sum_ec = ewt*(mrf[N-2,0] + mrf[N-1,1]) + cwt*mrf[N-2,1]
	sum_ec += lwt*img[N-1,0] - hwt
	prob = 1.0/(1.0 + exp(-beta*sum_ec))
	if prob < urand[N-1,0]: mrf[N-1,0] = 0
	else: mrf[N-1,0] = 1
	
	sum_ec = ewt*(mrf[N-2,P-1] + mrf[N-1,P-2]) + cwt*mrf[N-2,P-2]
	sum_ec += lwt*img[N-1,P-1] - hwt
	prob = 1.0/(1.0 + exp(-beta*sum_ec))
	if prob < urand[N-1,P-1]: mrf[N-1,P-1] = 0
	else: mrf[N-1,P-1] = 1


	#edges
	for j in range(1,P-2):
		sum_ec = ewt*(mrf[0,j-1] + mrf[0,j+1] + mrf[1,j])
		sum_ec += cwt*(mrf[1,j-1] + mrf[1,j+1])
		sum_ec += lwt*img[0,j] - hwt
		prob = 1.0/(1.0 + exp(-beta*sum_ec))
		if prob < urand[0,j]: mrf[0,j] = 0
		else: mrf[0,j] = 1

	
		sum_ec = ewt*(mrf[N-1,j-1] + mrf[N-1,j+1] + mrf[N-2,j])
		sum_ec += cwt*(mrf[N-2,j-1] + mrf[N-2,j+1])
		sum_ec += lwt*img[N-1,j] - hwt
		prob = 1.0/(1.0 + exp(-beta*sum_ec))
		if prob < urand[N-1,j]: mrf[N-1,j] = 0
		else: mrf[N-1,j] = 1

	for i in range(1,N-2):
		sum_ec = ewt*(mrf[i-1,0] + mrf[i+1,0] + mrf[i,1])
		sum_ec += cwt*(mrf[i-1,1] + mrf[i+1,1])
		sum_ec += lwt*img[i,0] - hwt
		prob = 1.0/(1.0 + exp(-beta*sum_ec))
		if prob < urand[i,0]: mrf[i,0] = 0
		else: mrf[i,0] = 1
	
		sum_ec = ewt*(mrf[i-1,P-1] + mrf[i+1,P-1] + mrf[i,P-2])
		sum_ec += cwt*(mrf[i-1,P-1] + mrf[i+1,P-1])
		sum_ec += lwt*img[i,P-1] - hwt
		prob = 1.0/(1.0 + exp(-beta*sum_ec))
		if prob < urand[i,P-1]: mrf[i,P-1] = 0
		else: mrf[i,P-1] = 1
	

	# middle
	for i in range(1,N-1):
		for j in range(1,P-1):
			sum_e = mrf[i,j-1] + mrf[i,j+1] + mrf[i-1,j] +mrf[i+1,j]
			sum_c = mrf[i-1,j-1] + mrf[i-1,j+1] + mrf[i+1,j-1] + mrf[i+1,j+1]  
			sum_ec = ewt*sum_e + cwt*sum_c + lwt*img[i,j] - hwt
			prob = 1.0/(1.0 + exp(-beta*sum_ec))
			if prob < urand[i,j]: mrf[i,j] = 0
			else: mrf[i,j] = 1


# This function is in place for error checking 
def parallel_ising_iter_gibbs(mrf,img,beta,ewt,cwt,lwt,hwt):
	parallel_ising_iter_gibbs_c(mrf,img,beta,ewt,cwt,lwt,hwt)

	return

@cython.boundscheck(False)
@cython.cdivision(True)
cdef parallel_ising_iter_gibbs_c(long[:,:] mrf, long[:,:] img,
		double beta, double ewt, double cwt, double lwt, double hwt):

	cdef int N = mrf.shape[0]
	cdef int P = mrf.shape[1]
	cdef int i,j
	cdef int start_idx
	cdef double sum_e, sum_c, sum_ec
	cdef double prob

	cdef double[:,:] urand = numpy.random.uniform(size=(N,P))


	# corners
	sum_ec = ewt*(mrf[0,1] + mrf[1,0]) + cwt*mrf[1,1]
	sum_ec += lwt*img[0,0] - hwt
	prob = 1.0/(1.0 + exp(-beta*sum_ec))
	if prob < urand[0,0]: mrf[0,0] = 0
	else: mrf[0,0] = 1
	
	sum_ec = ewt*(mrf[0,P-2] + mrf[1,P-1]) + cwt*mrf[1,P-2]
	sum_ec += lwt*img[0,P-1] - hwt
	prob = 1.0/(1.0 + exp(-beta*sum_ec))
	if prob < urand[0,P-1]: mrf[0,P-1] = 0
	else: mrf[0,P-1] = 1
	
	sum_ec = ewt*(mrf[N-2,0] + mrf[N-1,1]) + cwt*mrf[N-2,1]
	sum_ec += lwt*img[N-1,0] - hwt
	prob = 1.0/(1.0 + exp(-beta*sum_ec))
	if prob < urand[N-1,0]: mrf[N-1,0] = 0
	else: mrf[N-1,0] = 1
	
	sum_ec = ewt*(mrf[N-2,P-1] + mrf[N-1,P-2]) + cwt*mrf[N-2,P-2]
	sum_ec += lwt*img[N-1,P-1] - hwt
	prob = 1.0/(1.0 + exp(-beta*sum_ec))
	if prob < urand[N-1,P-1]: mrf[N-1,P-1] = 0
	else: mrf[N-1,P-1] = 1


	#edges
	for start_idx in range(1,3):
		for j in prange(start_idx,P-1,2,nogil=True):
			sum_e = ewt*(mrf[0,j-1] + mrf[0,j+1] + mrf[1,j])
			sum_c = cwt*(mrf[1,j-1] + mrf[1,j+1])
			sum_ec = sum_e + sum_c + lwt*img[0,j] - hwt
			prob = 1.0/(1.0 + exp(-beta*sum_ec))
			if prob < urand[0,j]: mrf[0,j] = 0
			else: mrf[0,j] = 1

	
			sum_e = ewt*(mrf[N-1,j-1] + mrf[N-1,j+1] + mrf[N-2,j])
			sum_c = cwt*(mrf[N-2,j-1] + mrf[N-2,j+1])
			sum_ec = sum_e + sum_c + lwt*img[N-1,j] - hwt
			prob = 1.0/(1.0 + exp(-beta*sum_ec))
			if prob < urand[N-1,j]: mrf[N-1,j] = 0
			else: mrf[N-1,j] = 1

	for start_idx in range(1,3):
		for i in prange(start_idx,N-1,nogil=True):
			sum_e = ewt*(mrf[i-1,0] + mrf[i+1,0] + mrf[i,1])
			sum_c = cwt*(mrf[i-1,1] + mrf[i+1,1])
			sum_ec = sum_e + sum_c + lwt*img[i,0] - hwt
			prob = 1.0/(1.0 + exp(-beta*sum_ec))
			if prob < urand[i,0]: mrf[i,0] = 0
			else: mrf[i,0] = 1
	
			sum_e = ewt*(mrf[i-1,P-1] + mrf[i+1,P-1] + mrf[i,P-2])
			sum_c = cwt*(mrf[i-1,P-1] + mrf[i+1,P-1])
			sum_ec = sum_e + sum_c + lwt*img[i,P-1] - hwt
			prob = 1.0/(1.0 + exp(-beta*sum_ec))
			if prob < urand[i,P-1]: mrf[i,P-1] = 0
			else: mrf[i,P-1] = 1
	

	# middle
	for start_idx in range(1,3):
		for i in prange(start_idx,N-1,2,nogil=True):
			for j in range(1,P-1):
				sum_e = mrf[i,j-1] + mrf[i,j+1] + mrf[i-1,j] +mrf[i+1,j]
				sum_c = mrf[i-1,j-1] + mrf[i-1,j+1] + mrf[i+1,j-1] + mrf[i+1,j+1]  
				sum_ec = ewt*sum_e + cwt*sum_c + lwt*img[i,j] - hwt
				prob = 1.0/(1.0 + exp(-beta*sum_ec))
				if prob < urand[i,j]: mrf[i,j] = 0
				else: mrf[i,j] = 1

