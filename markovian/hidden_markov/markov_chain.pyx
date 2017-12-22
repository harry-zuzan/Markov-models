import numpy 
cimport numpy

from libc.math cimport exp, sqrt
#from libc.math cimport M_PI

from likelihood import  gaussian_likelihood
from likelihood import  logistic_likelihood
from likelihood import  mixed_gaussian_likelihood
from likelihood import  mixed_logistic_likelihood

# These are the models with a Markov chain prior.  All priors are
# homogeneous and reversible.  Transitions probabilities are implemented
# in the form of a matrix with all diagonal elements equal and all
# off diagonal elements equal.

# The choices of likelihoods are Gaussian or logistic and mixtures of either
# to produce likelihods with heavier tails.

# There are two solutions. The Viterbi and marginal aka forward backward.


# The form of the prior and likelihood do not depend on each other.  In that
# sense they are always conjugate.  The approach is to compute the likelihood
# then feed it to the prior for a solution.


# ------------------------------------------------------------------

# In this section the likelihood is computed and fed to the solution 
# which implements the prior

def hmm_viterbi_gaussian(yvec, mu, sigma, diag):
	lhood = gaussian_likelihood(yvec, mu, sigma)

	soln = numpy.zeros((yvec.size,), numpy.int32)
	soln = hmm_viterbi_c(lhood, diag)

	return soln

def hmm_viterbi_logistic(yvec, mu, sigma, diag):
	lhood = logistic_likelihood(yvec, mu, sigma)

	soln = numpy.zeros((yvec.size,), numpy.int32)
	soln = hmm_viterbi_c(lhood, diag)

	return soln

def hmm_viterbi_mixed_gaussian(yvec, mu, sigma, cval, diag):
	lhood = mixed_gaussian_likelihood(yvec, mu, sigma, cval)

	soln = numpy.zeros((yvec.size,), numpy.int32)
	soln = hmm_viterbi_c(lhood, diag)

	return soln

def hmm_viterbi_mixed_logistic(yvec, mu, sigma, cval, diag):
	lhood = mixed_logistic_likelihood(yvec, mu, sigma, cval)

	soln = numpy.zeros((yvec.size,), numpy.int32)
	soln = hmm_viterbi_c(lhood, diag)

	return soln


def hmm_marginal_gaussian(yvec, mu, sigma, diag):
	lhood = gaussian_likelihood(yvec, mu, sigma)

	marginals = numpy.zeros((mu.size,yvec.size), numpy.float64)
	marginals = hmm_marginal_c(lhood, diag)
	soln = marginals.argmax(0)

	return marginals, soln


def hmm_marginal_logistic(yvec, mu, sigma, diag):
	lhood = logistic_likelihood(yvec, mu, sigma)

	marginals = numpy.zeros((mu.size,yvec.size), numpy.float64)
	marginals = hmm_marginal_c(lhood, diag)
	soln = marginals.argmax(0)

	return marginals, soln


def hmm_marginal_mixed_gaussian(yvec, mu, sigma, cval, diag):
	lhood = mixed_gaussian_likelihood(yvec, mu, sigma, cval)

	marginals = numpy.zeros((mu.size,yvec.size), numpy.float64)
	marginals = hmm_marginal_c(lhood, diag)
	soln = marginals.argmax(0)

	return marginals, soln


def hmm_marginal_mixed_logistic(yvec, mu, sigma, cval, diag):
	lhood = mixed_logistic_likelihood(yvec, mu, sigma, cval)

	marginals = numpy.zeros((mu.size,yvec.size), numpy.float64)
	marginals = hmm_marginal_c(lhood, diag)
	soln = marginals.argmax(0)

	return marginals, soln


# ------------------------------------------------------------------

# The Viterbi solution

cdef numpy.ndarray[numpy.int32_t,ndim=1] hmm_viterbi_c(
		numpy.ndarray[numpy.float64_t,ndim=2] lhood,
		double diag):

	cdef int N = lhood.shape[1]
	cdef int P = lhood.shape[0]

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
	for i in range(P): fwd[:,0] = lhood[:,0]
	fwd[:,0] /= fwd[:,0].sum()

	# compute the remaining forward probabilities keeping track of
	# the path to get to them
	# i is the source forward probability to transition from
	# j is the target forward probability to compute

	for t in range(1,N):
		for j in range(P):
			idx = 0
			fwd_trans_prob = 0.0
			for i in range(P):
				tmp_double = fwd[i,t-1]*trans[i,j]
				# if row i is the highest probability so far keep it
				if tmp_double > fwd_trans_prob:
					fwd_trans_prob = tmp_double
					idx = i

			# keep track of the path
			bwd[j,t-1] = idx
			
			# target is the most probable source times the likelihood
			fwd[j,t] = fwd_trans_prob*lhood[j,t]

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


# ------------------------------------------------------------------


cdef numpy.ndarray[numpy.float64_t,ndim=2] hmm_marginal_c(
		numpy.ndarray[numpy.float64_t,ndim=2] lhood,
		double diag):

	cdef int N = lhood.shape[1]
	cdef int P = lhood.shape[0]

	cdef int i,j,t
	cdef int idx

	cdef double tmp_double;

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
	fwd[:,0] = lhood[:,0]/lhood[:,0].sum()

	# compute the remaining forward probabilities
	for t in range(1,N):
		# j is the target forward probability to compute
		for j in range(P):
			# add up the transitions
			for i in range(P): fwd[j,t] += fwd[i,t-1]*trans[i,j]
			
			# multiply by the likelihood
			fwd[j,t] *= lhood[j,t]

		# normalise the probabilities
		fwd[:,t] /= fwd[:,t].sum()

	# the last backward probability is based on no data following
	bwd[:,N-1] = 1.0/P

	# the remaining are the probability of the state given all future
	# data computed recursively

	t = N-2
	while t >= 0:
		for j in range(P):
			# probability of the hidden state going through (j,t+1)
			tmp_double = bwd[j,t+1]*lhood[j,t+1]
			for i in range(P):
				bwd[i,t] += trans[i,j]*tmp_double
		bwd[:,t] /= bwd[:,t].sum()
		t -= 1

	marginals = fwd*bwd
	marginals /= marginals.sum(0)

	return marginals
