##import pywt
import numpy
cimport numpy

from libc.math cimport exp, sqrt
#from libc.math cimport M_PI

# gaussian probabilities not fully normalised since they are used
# for likelihoods
def prob_normal_vec(vec, mu, sigma):
	vec1 = (vec - mu)/sigma
	return numpy.exp(-0.5*vec1*vec1)/(sigma*numpy.sqrt(numpy.pi))

def prob_logistic_vec(vec, mu, sigma):
	s = sigma*numpy.sqrt(3.0)/numpy.pi
	vec1 = (vec - mu)/s

	numerator = numpy.exp(-vec1)
	denominator = s*(1.0 + numpy.exp(-vec1))**2

	return numerator/denominator

# may want to introduce scalar versions at a later time for speed
# in the meantime because it is pure Python scalar in => scalar out
prob_normal_scalar = prob_normal_vec
prob_logistic_scalar = prob_logistic_vec

# Likelihoods computed below
def gaussian_likelihood(yvec, mu, sigma):
	lhood = numpy.zeros((mu.size,yvec.size), numpy.float64)

	for i in range(mu.size):
		lhood[i,:] = prob_normal_vec(yvec,mu[i],sigma[i])
	lhood /= lhood.sum(0)

	return lhood

def logistic_likelihood(yvec, mu, sigma):
	lhood = numpy.zeros((mu.size,yvec.size), numpy.float64)

	for i in range(mu.size):
		lhood[i,:] = prob_logistic_vec(yvec,mu[i],sigma[i])
	lhood /= lhood.sum(0)

	return lhood


# Note that the method by Chipman is more complicated
def mixed_gaussian_likelihood(yvec, mu, sigma, cval):
	lhood = numpy.zeros((mu.size,yvec.size), numpy.float64)

	for i in range(mu.size):
		prob_vec1 = prob_normal_vec(yvec,mu[i],sigma[i])
		prob_vec2 = prob_normal_vec(yvec,mu[i],sigma[i]*cval)

		prob_lhood1 = prob_vec1/(prob_vec1 + prob_vec2)

		# fix up the weights so they do not redescend at zero
		prob1 = prob_normal_scalar(0.0, 0.0, sigma[i])
		prob2 = prob_normal_scalar(0.0, 0.0, sigma[i]*cval)

		prob_lhood1 /= prob1/(prob1 + prob2)

		lhood[i,:] = prob_lhood1*prob_vec1 + (1.0 - prob_lhood1)*prob_vec2

	lhood /= lhood.sum(0)

	return lhood



def mixed_logistic_likelihood(yvec, mu, sigma, cval):
	lhood = numpy.zeros((mu.size,yvec.size), numpy.float64)

	for i in range(mu.size):
		prob_vec1 = prob_logistic_vec(yvec,mu[i],sigma[i])
		prob_vec2 = prob_logistic_vec(yvec,mu[i],sigma[i]*cval)

		prob_lhood1 = prob_vec1/(prob_vec1 + prob_vec2)

		# fix up the weights so they do not redescend at zero
		prob1 = prob_logistic_scalar(0.0, 0.0, sigma[i])
		prob2 = prob_logistic_scalar(0.0, 0.0, sigma[i]*cval)

		prob_lhood1 /= prob1/(prob1 + prob2)

		lhood[i,:] = prob_lhood1*prob_vec1 + (1.0 - prob_lhood1)*prob_vec2

	lhood /= lhood.sum(0)

	return lhood

