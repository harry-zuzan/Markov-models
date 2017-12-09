##import pywt
import numpy 
cimport numpy

from libc.math cimport exp, sqrt
#from libc.math cimport M_PI


# the probabilities of observed values in a zero mean standard distribution
## with standard deviation.
## default is standard normal
#def prob_normal_vec_1d(vec, s=1.0):
#	return c_prob_normal_vec_1d(vec.astype(numpy.double), numpy.double(s))
#
#

# gaussian probabilities sans the normalisation constant
cdef double prob_normal(double val,double mu,double sigma):
	cdef double val1 = (val - mu)/sigma
	return exp(-0.5*val1*val1)/sigma


def hmm_viterbi_gaussian(yvec, mu, sigma, diag):
	return c_hmm_viterbi_gaussian(yvec, mu, sigma, diag)


cdef numpy.ndarray[numpy.int32_t,ndim=1] c_hmm_viterbi_gaussian(
		numpy.ndarray[numpy.float64_t,ndim=1] yvec,
		numpy.ndarray[numpy.float64_t,ndim=1] mu,
		numpy.ndarray[numpy.float64_t,ndim=1] sigma,
		double diag):

	cdef int N = yvec.size
	cdef int P = mu.size

	cdef int i,j,t
	cdef int idx

	cdef double fwd_trans_prob;
	cdef double tmp_double;

	# the solution
	cdef numpy.ndarray[numpy.int32_t,ndim=1] soln = \
			numpy.zeros((N,), numpy.int32)

	# forward probabilities
	cdef numpy.ndarray[numpy.float64_t,ndim=2] fwd = \
			numpy.zeros((P,N), numpy.double)

	# backward path through the MAP/Viterbi soln
	cdef numpy.ndarray[numpy.int32_t,ndim=2] bwd = \
			numpy.zeros((P,N), numpy.int32)

	# transition probabilities in the form of a matrix
	cdef numpy.ndarray[numpy.float64_t,ndim=2] trans = \
			numpy.zeros((P,P), numpy.double)

	# fill the transition matrix with the off diagonal
	trans.fill((1.0 - diag)/(P - 1))

	# overwrite the diagonal 
	for i in range(P): trans[i,i] = diag


	# here the Viterbi algorithm proceeds forward probabilities
	# are normalised to sum to one for numerical stability

	# the first forward probability is likelihood only
	for i in range(P):
		fwd[i,0] = prob_normal(yvec[0],mu[i],sigma[i])
	fwd[:,0] /= fwd[:,0].sum()

	# compute the remaining forward probabilities keeping track of
	# the path to get to them
	for t in range(1,N):
		# j is the target forward probability to compute
		for j in range(P):
			idx = 0
			fwd_trans_prob = 0.0
			# i is the source forward probability to transition from
			for i in range(P):
				tmp_double = fwd[i,t-1]*trans[i,j]
				# if row i is the highest probability so far keep it
				if tmp_double > fwd_trans_prob:
					fwd_trans_prob = tmp_double
					idx = i

			# keep track of the path
			bwd[j,t-1] = idx
			
			# target is the most probable source times the likelihood
			fwd[j,t] = fwd_trans_prob*prob_normal(yvec[t],mu[j],sigma[j])

		# normalise the probabilities
		fwd[:,t] /= fwd[:,t].sum()

	# the most probable aka MAP aka Viterbi path ends at N-1 with
	# greater probability than any other
	soln[N-1] = fwd[:,N-1].argmax()


	# work through the path backwards from soln[N-1]
	t = N-1
	while t > 0:
		soln[t-1] = bwd[soln[t],t-1]
		t -= 1

	return soln


def hmm_marginal_gaussian(yvec, mu, sigma, diag):
	return c_hmm_marginal_gaussian(yvec, mu, sigma, diag)


cdef numpy.ndarray[numpy.int32_t,ndim=1] c_hmm_marginal_gaussian(
		numpy.ndarray[numpy.float64_t,ndim=1] yvec,
		numpy.ndarray[numpy.float64_t,ndim=1] mu,
		numpy.ndarray[numpy.float64_t,ndim=1] sigma,
		double diag):

	cdef int N = yvec.size
	cdef int P = mu.size

	cdef int i,j,t
	cdef int idx

	cdef double tmp_double;

	# the solution
	cdef numpy.ndarray[numpy.int32_t,ndim=1] soln = \
			numpy.zeros((N,), numpy.int32)

	# forward probabilities
	cdef numpy.ndarray[numpy.float64_t,ndim=2] fwd = \
			numpy.zeros((P,N), numpy.double)

	# backward probabilities
	cdef numpy.ndarray[numpy.float64_t,ndim=2] bwd = \
			numpy.zeros((P,N), numpy.double)

	# transition probabilities in the form of a matrix
	cdef numpy.ndarray[numpy.float64_t,ndim=2] trans = \
			numpy.zeros((P,P), numpy.double)

	# fill the transition matrix with the off diagonal
	trans.fill((1.0 - diag)/(P - 1))

	# overwrite the diagonal 
	for i in range(P): trans[i,i] = diag


	# the marginal algorithm forward probabilities are normalised to
	# sum to one for numerical stability

	# the first forward probability is likelihood only
	for i in range(P):
		fwd[i,0] = prob_normal(yvec[0],mu[i],sigma[i])
	fwd[:,0] /= fwd[:,0].sum()

	# compute the remaining forward probabilities
	for t in range(1,N):
		# j is the target forward probability to compute
		for j in range(P):
			# add up the transitions
			for i in range(P): fwd[j,t] += fwd[i,t-1]*trans[i,j]
			
			# multiply by the likelihood
			fwd[j,t] *= prob_normal(yvec[t],mu[j],sigma[j])

		# normalise the probabilities
		fwd[:,t] /= fwd[:,t].sum()

	# the last backward probability is based on no data following
	for j in range(P): bwd[j,N-1] = 1.0/P

	# the remaining are the probability of the state given all future
	# data computed recursively

	t = N-2
	while t >= 0:
		for j in range(P):
			# probability of the hidden state going through (j,t+1)
			tmp_double = bwd[j,t+1]*prob_normal(yvec[t+1],mu[j],sigma[j])
			for i in range(P):
				bwd[i,t] += trans[i,j]*tmp_double
		bwd[:,t] /= bwd[:,t].sum()
		t -= 1

	soln = (fwd*bwd).argmax(0).astype(numpy.int32)

	return soln



#	c_prob_normal_vec_1d(numpy.ndarray[numpy.float64_t,ndim=1] vec, double s):
#
#	cdef numpy.ndarray[numpy.float64_t,ndim=1] prob_vec = \
#			numpy.exp(-0.5*(vec/s)**2)
#
#	return prob_vec/(s*sqrt(2.0*M_PI))
#

#cdef numpy.ndarray[numpy.float64_t,ndim=1] \
#	c_prob_normal_vec_1d(numpy.ndarray[numpy.float64_t,ndim=1] vec, double s):
#
#	cdef numpy.ndarray[numpy.float64_t,ndim=1] prob_vec = \
#			numpy.exp(-0.5*(vec/s)**2)
#
#	return prob_vec/(s*sqrt(2.0*M_PI))
#
#
#def get_prob_no_signal(resids, cval, stdev=None):
#	# probably want to use MAD instead
#	resids = resids.astype(numpy.double)
#	cval = numpy.double(cval)
#	if stdev is None: stdev = resids.std()
#
#	return c_prob_no_signal(resids, cval, stdev)
#
#
#cdef numpy.ndarray[numpy.float64_t,ndim=1] \
#	c_prob_no_signal(numpy.ndarray[numpy.float64_t,ndim=1] vec,
#		double cval, double stdev):
#
#	cdef numpy.ndarray[numpy.float64_t,ndim=1] prob_vec1 = \
#			c_prob_normal_vec_1d(vec, stdev)
#
#
#	cdef numpy.ndarray[numpy.float64_t,ndim=1] prob_vec2 = \
#			c_prob_normal_vec_1d(vec, cval*stdev)
#
#
#	cdef numpy.ndarray[numpy.float64_t,ndim=1] prob_no_signal = \
#			prob_vec1/(prob_vec1 + prob_vec2)
#
#	return prob_no_signal
#
#
#def forward(eprob,diag):
#	return c_forward(eprob,diag)
#
#cdef numpy.ndarray[numpy.float64_t,ndim=2] \
#	c_forward(numpy.ndarray[numpy.float64_t,ndim=1] eprob, double diag):
#
#	cdef int N = eprob.shape[0]
#
#	cdef numpy.ndarray[numpy.float64_t,ndim=2] fwd_probs = \
#		numpy.zeros((2,N), numpy.double)
#
#	fwd_probs[0,0] = eprob[0]
#	fwd_probs[1,0] = 1.0 - eprob[0]
#
#	cdef int i
#	cdef double prob_sum
#
#	for i from 0 < i < N:
#		fwd_probs[0,i] = fwd_probs[0,i-1]*diag*eprob[i]
#		fwd_probs[0,i] += fwd_probs[1,i-1]*(1.0 - diag)*eprob[i]
#
#		fwd_probs[1,i] = fwd_probs[1,i-1]*diag*(1.0 - eprob[i])
#		fwd_probs[1,i] += fwd_probs[0,i-1]*(1.0 - diag)*(1.0 - eprob[i])
#
#		prob_sum = fwd_probs[0,i] + fwd_probs[1,i]
#		fwd_probs[0,i] /= prob_sum
#		fwd_probs[1,i] /= prob_sum
#
#	return fwd_probs
#
#
#def backward(eprob,diag):
#	return c_backward(eprob,diag)
#
#
#cdef numpy.ndarray[numpy.float64_t,ndim=2] \
#	c_backward(numpy.ndarray[numpy.float64_t,ndim=1] eprob, double diag):
#
#	cdef int N = eprob.shape[0]
#
#	cdef numpy.ndarray[numpy.float64_t,ndim=2] bwd_probs = \
#		numpy.zeros((2,N), numpy.double)
#
#	cdef int i
#	cdef double prob_sum
#
#	bwd_probs[0,N-1] = 0.5
#	bwd_probs[1,N-1] = 0.5
#
#	for i from N-1 > i >= 0:
#		bwd_probs[0,i] = bwd_probs[0,i+1]*diag*eprob[i+1]
#		bwd_probs[0,i] += bwd_probs[1,i+1]*(1.0 - diag)*(1.0 - eprob[i+1])
#
#		bwd_probs[1,i] = bwd_probs[1,i+1]*diag*(1.0 - eprob[i+1])
#		bwd_probs[1,i] += bwd_probs[0,i+1]*(1.0 - diag)*eprob[i+1]
#
#		bwd_probs[:,i] /= bwd_probs[:,i].sum()
#
#	return bwd_probs


# the most probable solution given gaussian emissions
# use this to fix up data types for the pyx code then call
