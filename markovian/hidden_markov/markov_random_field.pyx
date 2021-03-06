# Need to document

import numpy 
cimport numpy
cimport cython

from libc.math cimport exp, sqrt
#from libc.math cimport M_PI

from likelihood import  gaussian_likelihood
from likelihood import  logistic_likelihood

from likelihood import  mixed_gaussian_likelihood
from likelihood import  mixed_logistic_likelihood


# Put the likelihoods in place
def hmrf_gaussian(yvec, mu, sigma, diag, converge=10.0**-6,max_iter=64):
	lhood = gaussian_likelihood(yvec, mu, sigma)
	return solve_hmrf(lhood, diag, converge, max_iter)

def hmrf_mixed_gaussian(yvec, mu, sigma, cval, diag,
		converge=10.0**-6,max_iter=64):
	lhood = mixed_gaussian_likelihood(yvec, mu, sigma, cval)
	return solve_hmrf(lhood, diag, converge, max_iter)

def hmrf_logistic(yvec, mu, sigma, diag, converge=10.0**-6,max_iter=64):
	lhood = logistic_likelihood(yvec, mu, sigma)
	return solve_hmrf(lhood, diag, converge, max_iter)

def hmrf_mixed_logistic(yvec, mu, sigma, cval, diag,
		converge=10.0**-6,max_iter=64):
	lhood = mixed_logistic_likelihood(yvec, mu, sigma, cval)
	return solve_hmrf(lhood, diag, converge, max_iter)


def hmrf_gaussian_sample(yvec, mu, sigma, diag, nburn,nsample):
	lhood = gaussian_likelihood(yvec, mu, sigma)
	return sample_hmrf(lhood, diag, nburn, nsample)


# Execute the solution given the likelihood
def	solve_hmrf(lhood, diag, converge, max_iter):
	P,N = lhood.shape

	# the Markov random field
	mrf = numpy.zeros((P,N), numpy.float64)
	mrf.fill(1.0/P)

	# transition probabilities in the form of a matrix
	trans = numpy.zeros((P,P), numpy.double)
	trans.fill((1.0 - diag)/(P - 1))
	for i in range(P): trans[i,i] = diag

	idx = 0
	while idx < max_iter:
		converge_iter = hmrf_iter_cy(mrf, lhood, trans)
		if converge_iter < converge: break

		idx += 1

	print 'converged at iteration', idx

	soln = mrf.argmax(0)

	return mrf, soln


# one iteration of the function to solve the mrf given data
@cython.boundscheck(False)
cdef double hmrf_iter_cy(
		numpy.ndarray[numpy.float64_t,ndim=2] mrf,
		numpy.ndarray[numpy.float64_t,ndim=2] lhood,
		numpy.ndarray[numpy.float64_t,ndim=2] trans):


	cdef int N = lhood.shape[1]
	cdef int P = lhood.shape[0]

	cdef int i,j,t

	cdef double converge_sum

	cdef double prob_left, prob_right;

	# hold temporary values to test for convergence
	cdef numpy.ndarray[numpy.float64_t,ndim=1] mrf_tmp = \
			numpy.zeros((P,), numpy.double)

	# Probably should deal with the end points as if they have missing
	# data on either side.
	
	converge_sum = 0.0
	# left end point
	mrf_tmp[:] = mrf[:,0]
	for i in range(P):
		prob_right = 0.0
		for j in range(P): prob_right += trans[i,j]*mrf[j,1]
		mrf[i,0] = prob_right*lhood[i,0]
	mrf[:,0] /= mrf[:,0].sum()

	mrf_tmp -= mrf[:,0]
	converge_sum += (mrf_tmp*mrf_tmp).sum()

	# right end point
	mrf_tmp[:] = mrf[:,N-1]
	for j in range(P):
		prob_left = 0.0
		for i in range(P): prob_left += trans[i,j]*mrf[i,N-2]
		mrf[j,N-1] = prob_left*lhood[j,N-1]
	mrf[:,N-1] /= mrf[:,N-1].sum()

	mrf_tmp -= mrf[:,N-1]
	converge_sum += (mrf_tmp*mrf_tmp).sum()

	# the middle
	for t in range(1,N-1):
		mrf_tmp[:] = mrf[:,t]
		for j in range(P):
			prob_left = 0.0
			prob_right = 0.0
			for i in range(P):
				prob_left += trans[i,j]*mrf[i,t-1]
				prob_right += trans[i,j]*mrf[i,t+1]
			mrf[j,t] = prob_left*prob_right*lhood[j,t]
		mrf[:,t] /= mrf[:,t].sum()

		mrf_tmp -= mrf[:,t]
		converge_sum += (mrf_tmp*mrf_tmp).sum()

	converge_sum /= N

	return converge_sum



# Execute the solution given the likelihood
def	sample_hmrf(lhood, diag, nburn, nsample):
	P,N = lhood.shape

	# the Markov random field
	mrf = numpy.zeros((P,N), numpy.float64)
	mrf.fill(1.0/P)

	# transition probabilities in the form of a matrix
	trans = numpy.zeros((P,P), numpy.double)
	trans.fill((1.0 - diag)/(P - 1))
	for i in range(P): trans[i,i] = diag

	hmrf_gaussian_gibbs_cy(mrf,lhood,trans,nburn,nsample)

	soln = mrf.argmax(0)

	return mrf, soln



# Gibbs samles the states - either the mrf is burned in or use this to burn in
@cython.boundscheck(False)
cdef void hmrf_gaussian_gibbs_cy(
		numpy.ndarray[numpy.float64_t,ndim=2] mrf,
		numpy.ndarray[numpy.float64_t,ndim=2] lhood,
		numpy.ndarray[numpy.float64_t,ndim=2] trans,
		int nburn, int nsample):


	cdef int N = lhood.shape[1]
	cdef int P = lhood.shape[0]

	cdef int i,j,t,idx,iter

	cdef double state_sum;

	# keeps track of the number of times a state is realised
	cdef numpy.ndarray[numpy.int64_t,ndim=2] mrf_count = \
			numpy.zeros((P,N), numpy.int64)

	# keeps track of the current state initialized with the max of mrf
	cdef numpy.ndarray[numpy.int64_t,ndim=1] mrf_state = mrf.argmax(0)

	# uniformly distributed random numbers for Gibbs sampling
	cdef numpy.ndarray[numpy.float64_t,ndim=1] rval = \
			numpy.zeros((N,),numpy.double)


	print('number of iterations =', nsample)

	for iter in range(nburn + nsample):
		rval = numpy.random.uniform(size=N)
	
		# left end point
		for i in range(P): mrf[i,0] = trans[i,mrf_state[0]]*lhood[i,0]
		mrf[:,0] /= mrf[:,0].sum()

		idx = 0
		state_sum = mrf[0,0]

		while idx < P:
			if rval[0] < state_sum: break
			idx += 1
			state_sum += mrf[idx,0]
		mrf_state[0] = idx



		# right end point
		for i in range(P): mrf[i,N-1] = trans[i,mrf_state[N-2]]*lhood[i,N-1]
		mrf[:,N-1] /= mrf[:,N-1].sum()

		idx = 0
		state_sum = mrf[0,N-1]

		while idx < P:
			if rval[N-1] < state_sum: break
			idx += 1
			state_sum += mrf[idx,N-1]
		mrf_state[N-1] = idx

		# the middle
		for t in range(1,N-1):
			for i in range(P):
				mrf[i,t] = trans[i,mrf_state[t-1]]*trans[i,mrf_state[t+1]]
			mrf[:,t] *= lhood[:,t]
			mrf[:,t] /= mrf[:,t].sum()

			idx = 0
			state_sum = mrf[0,t]

			while idx < P:
				if rval[t] < state_sum: break
				idx += 1
				state_sum += mrf[idx,t]
			mrf_state[t] = idx

		if iter < nburn: continue

		for t in range(N): mrf_count[mrf_state[t],t] += 1


	mrf = mrf_count.astype(numpy.double)/nsample

