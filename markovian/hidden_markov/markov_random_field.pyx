import numpy 
cimport numpy

from libc.math cimport exp, sqrt
#from libc.math cimport M_PI

# gaussian probabilities sans the normalisation constant
cdef double prob_normal(double val,double mu,double sigma):
	cdef double val1 = (val - mu)/sigma
	return exp(-0.5*val1*val1)/sigma


def hmrf_gaussian(yvec, mu, sigma, diag, converge=10.0**-6,max_iter=64):
	return c_hmrf_gaussian(yvec, mu, sigma, diag, converge, max_iter)


# one iteration of the function to solve the mrf given data
cdef double c_solve_mrf_iter(
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


# function to solve the mrf given data
cdef numpy.ndarray[numpy.float64_t,ndim=2] c_solve_mrf(
		numpy.ndarray[numpy.float64_t,ndim=2] lhood,
		numpy.ndarray[numpy.float64_t,ndim=2] trans,
		double converge, int max_iter):


	cdef int N = lhood.shape[1]
	cdef int P = lhood.shape[0]

	cdef int idx

	cdef double converge_iter

	# Markov random field
	cdef numpy.ndarray[numpy.float64_t,ndim=2] mrf = \
			numpy.zeros_like(lhood)

	# initialise the mrf
	mrf.fill(1.0/P)

	idx = 0
	while idx < max_iter:
		converge_iter = c_solve_mrf_iter(mrf, lhood, trans)
		print 'converge_iter =', converge_iter
		if converge_iter < converge: break

		idx += 1

	return mrf



# start breaking up the code in to functions
cdef numpy.ndarray[numpy.int32_t,ndim=1] c_hmrf_gaussian(
		numpy.ndarray[numpy.float64_t,ndim=1] yvec,
		numpy.ndarray[numpy.float64_t,ndim=1] mu,
		numpy.ndarray[numpy.float64_t,ndim=1] sigma,
		double diag, double converge, int max_iter):

	cdef int N = yvec.size
	cdef int P = mu.size

	cdef int i,t

	# the solution
	cdef numpy.ndarray[numpy.int32_t,ndim=1] soln = \
			numpy.zeros((N,), numpy.int32)

	# Markov random field
	cdef numpy.ndarray[numpy.float64_t,ndim=2] mrf = \
			numpy.zeros((P,N), numpy.float64)

	# Likelihood
	cdef numpy.ndarray[numpy.float64_t,ndim=2] lhood = \
			numpy.zeros((P,N), numpy.float64)

	# transition probabilities in the form of a matrix
	cdef numpy.ndarray[numpy.float64_t,ndim=2] trans = \
			numpy.zeros((P,P), numpy.double)

	# fill the transition matrix with the off diagonal
	trans.fill((1.0 - diag)/(P - 1))

	# overwrite the diagonal 
	for i in range(P): trans[i,i] = diag

	print 'inside c_hmrf_gaussian'

	# fill out the likelihood
	for t in range(N):
		for i in range(P):
			lhood[i,t] = prob_normal(yvec[t],mu[i],sigma[i])
	lhood /= lhood.sum(0)

	mrf = c_solve_mrf(lhood,trans,converge,max_iter)

	soln = mrf.argmax(0).astype(numpy.int32)

	return soln
