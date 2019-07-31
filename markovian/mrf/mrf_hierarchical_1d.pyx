# Need to document

import numpy as np
cimport numpy as cnp
cimport cython

class QTree1:
	def __init__(self,double[:] obs):
		# Root at this point is the highest level not the access point
		self.root = None

		# Leaves are the lowest level and the only level that contain data
		# That is, likelihood only applies to the leaves
		self.leafs = None

		# The observed data
		self.obs = np.asarray(obs.copy())

		# For now use a list to store the resolutions
		self.levels = []

		# Since the number of observations is not a power of 2 keep track
		# of the lengths
		self.level_lengths = []

		# Build the tree by recursively inserting at the top the next
		# resolution and its length
		N = self.obs.size
		level_vec = self.obs.copy()
		self.levels.insert(0,level_vec)
		self.level_lengths.insert(0,N)

		# push the lower resolutions to the front until the root
		while 1:
			print('N =', N)

			# if the length of the previous resolution is even it divides
			# by 2 so the end is handled differently than if it is odd
			odd_length = False
			if N%2: odd_length = True

			# the next lower resolution
			N = N//2 + N%2
			level_vec1 = np.zeros((N,),np.double)

			# average the two from the lower level stopping one short of
			# the end which may encounter the odd number problem
			for i in range(N-1):
				level_vec1[i] = 0.5*(level_vec[2*i] + level_vec[2*i + 1])

			# assume it is odd in which case it just takes on the value
			# of the straggler
			level_vec1[N-1] = level_vec[2*(N-1)]

			# but if it is even it gets averaged after including the end
			if not odd_length:
				level_vec1[N-1] += level_vec[2*(N-1) + 1]
				level_vec1[N-1] *= 0.5

			# keep a reference and insert it into the front of the list 
			level_vec = level_vec1.copy()
			self.levels.insert(0,level_vec)
			self.level_lengths.insert(0,N)

			# finished if the root has been reached
			if N == 1: break

		# set the root
		self.root = level_vec

		print('**N =', N)



	def sweep_down(self):
		K = len(self.levels)
		if K < 2: return

		self.levels[0][0] = 0.5*(self.levels[1][0] + self.levels[1][1])

		for k in range(1,K-1):
			self.levels[k][0]  = 4.0*self.levels[k-1][0]
			self.levels[k][0] += 2.0*self.levels[k][1]
			self.levels[k][0] += self.levels[k+1][0] + self.levels[k+1][1] 
			self.levels[k][0] /= 8.0

			J = self.level_lengths[k]
			for j in range(1,J-1):
				self.levels[k][j]  = 4.0*self.levels[k-1][j//2]
				self.levels[k][j] += 2.0*self.levels[k][j-1]
				self.levels[k][j] += 2.0*self.levels[k][j+1]
				self.levels[k][j] += self.levels[k+1][2*j]
				self.levels[k][j] += self.levels[k+1][2*j + 1]
				self.levels[k][j] /= 10.0

			self.levels[k][J-1]  = 4.0*self.levels[k-1][(J-1)//2]
			self.levels[k][J-1] += 2.0*self.levels[k][J-2]
			self.levels[k][J-1] += self.levels[k+1][2*(J-1)]

			if self.level_lengths[k+1] % 2:
				self.levels[k][J-1] /= 7.0
				continue

			self.levels[k][J-1] += self.levels[k+1][2*(J-1) + 1]
			self.levels[k][J-1] /= 8.0


	def mrf_leafs(self, pprec, lprec):
		if len(self.levels) < 2: return

		J = len(self.level_lengths[-1])

		self.levels[-1][0] = pprec*(2.0*self.levels[-2][0] + self.levels[-1][1])
		self.levels[-1][0] += lprec*self.obs[0]
		self.levels[-1][0] /= 3.0*pprec + lprec

		self.levels[-1][J-1]  = 2.0*self.levels[-2][(J-1)//2]
		self.levels[-1][J-1] += self.levels[-1][J-2]
		self.levels[-1][J-1] *= pprec
		self.levels[-1][J-1] += lprec*self.obs[J-1]
		self.levels[-1][J-1] /= 3*pprec + lprec


		for j in range(1,J-1):
			self.levels[-1][j] = 2.0*self.levels[-2][j//2]
			self.levels[-1][j] += self.levels[-1][j-1] + self.levels[-1][j+1]
			self.levels[-1][j] *= pprec
			self.levels[-1][j] += lprec*self.obs[j]
			self.levels[-1][j] /= 4.0*pprec + lprec

		

				


# Execute the solution given the likelihood
@cython.boundscheck(False)
def shrink_mrf1_harch(double[:] obs, double eprec, double lprec,
		double converged = 1e-6, max_iter=1024):
	cdef int N = obs.shape[0]

	# the Markov random field
	cdef double[:]  mrf = obs.copy()

	cdef int iter

	cdef double diff

#	qtree = QTree1(obs)

	iter = 0
	while 1:
		iter = iter + 1
		if iter > max_iter: break
		diff = mrf1_harch_iter(obs, mrf, eprec, lprec)
		if diff < converged: break

	print('iter,diff =', iter,diff)

	return np.asarray(mrf)



# Execute the solution given the likelihood
@cython.boundscheck(False)
cdef mrf1_harch_iter(double[:] obs, double[:] mrf, double eprec, double lprec):
	cdef int N = obs.shape[0]

	# the Markov random field
	cdef double[:]  mrf_old = mrf.copy()
	cdef double[:]  diffs = np.zeros_like(mrf)

	cdef int i

	cdef double prec = eprec + lprec

	mrf[0] = eprec*mrf[1] + lprec*obs[0]
	mrf[0] = mrf[0]/prec

	diffs[0] = mrf[0] - mrf_old[0]

	mrf[N-1] = eprec*mrf[N-2] + lprec*obs[N-1]
	mrf[N-1] = mrf[N-1]/prec

	diffs[N-1] = mrf[N-1] - mrf_old[N-1]

	prec = 2.0*eprec + lprec

	for i in range(1,N-1):
		mrf[i] = eprec*(mrf[i-1] + mrf[i+1]) + lprec*obs[i]
		mrf[i] = mrf[i]/prec
		diffs[i] = mrf[i] - mrf_old[i]

	return np.abs(diffs).max()

