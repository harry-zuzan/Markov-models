import numpy as np
cimport numpy as cnp
import cython
from cython.parallel import prange, parallel

# should be cimport to run in parallel
#from libc.math import exp as exp_c
from libc.math cimport fabs as fabs_c

from redescendl import redescend_normal3

#--------------------------------------------------------------------

#@cython.boundscheck(False)
#@cython.cdivision(True)
def shrink_mrf3_icm(double[:,:,:] obs,
			double prior_side_prec, double prior_edge_prec,
			double prior_diag_prec, double likelihood_prec,
			double relax_coeff=0.35, double converged=1e-6):

	print('here 1')

	cdef double[:,:,:] shrunk = obs.copy()

	cdef int relax = 0

	iter = 1
	while 1:
		diff = shrink_mrf3_icm_iter(obs, shrunk,
					prior_side_prec, prior_edge_prec,
					prior_diag_prec, likelihood_prec,
					relax, relax_coeff)

		print('here iter', iter, diff)
		if diff < converged: break

		iter += 1
		if iter > 3: relax = 1

	print('here 2')

	return np.asarray(shrunk)


@cython.boundscheck(False)
@cython.cdivision(True)
def shrink_mrf3_redescend(double[:,:,:] obs,
			double prior_side_prec, double prior_edge_prec,
			double prior_diag_prec, double likelihood_prec,
			double cval, int max_iter=30, double converged=1e-6):

	print('here 1')

	cdef int M = obs.shape[0]
	cdef int N = obs.shape[1]
	cdef int P = obs.shape[2]

	cdef int i,j,k

	cdef double[:,:,:] redesc = obs.copy()
	cdef double[:,:,:] shrunk = obs.copy()
	cdef double[:,:,:] shrunk_old = np.zeros_like(obs)
	cdef double[:,:,:] resids = np.zeros_like(obs)

	cdef double[:,:,:] diffs = np.zeros_like(obs)
	cdef double diff

	cdef int iter, idx

#	cdef int relax = 0

	iter = 0
	while 1:
		iter += 1
#		if iter > max_iter: break

		shrunk_old = shrunk.copy()

		diff = shrink_mrf3_icm_iter(redesc, shrunk,
					prior_side_prec, prior_edge_prec,
					prior_diag_prec, likelihood_prec)

		for i in prange(M,nogil=True):
			for j in range(N):
				for k in range(P):
					diffs[i,j,k] = shrunk[i,j,k] - shrunk_old[i,j,k]

		diff = np.abs(diffs).max()
		print('iter, diff =', iter + 1, diff)
		if diff < converged: break

		if iter > 3:
			for i in prange(M,nogil=True):
				for j in range(N):
					for k in range(P):
						shrunk[i,j,k] = shrunk[i,j,k] + 0.35*diffs[i,j,k]

		for i in prange(M,nogil=True):
			for j in range(N):
				for k in range(P):
					resids[i,j,k] = obs[i,j,k] - shrunk[i,j,k]

		resids = redescend_normal3(np.asarray(resids),cval)

		for i in prange(M,nogil=True):
			for j in range(N):
				for k in range(P):
					redesc[i,j,k] = resids[i,j,k] + shrunk[i,j,k]



	print('here 2')

	return np.asarray(shrunk)


@cython.boundscheck(False)
@cython.cdivision(True)
def shrink_mrf3_icm_iter(double[:,:,:] obs, double[:,:,:] shrunk,
			double prior_side_prec, double prior_edge_prec,
			double prior_diag_prec, double likelihood_prec,
			int relax=0, double relax_coeff=0.35):

	cdef int M = obs.shape[0]
	cdef int N = obs.shape[1]
	cdef int P = obs.shape[2]
	cdef double[:,:,:] diffs = shrunk.copy()

	cdef int i, j, k
	cdef double prec, val
	
	# just to make the code a bit more compact
	cdef double sprec = prior_side_prec
	cdef double eprec = prior_edge_prec
	cdef double dprec = prior_diag_prec
	cdef double lprec = likelihood_prec


	# for staggering in parallel applications
	cdef int idx

	
	# corners
	prec = 3.0*sprec + 3.0*eprec + dprec + lprec

	# top left corner on the lower face is at voxel 0,0,0
	val = sprec*(shrunk[1,0,0] + shrunk[0,1,0] + shrunk[0,0,1])
	val += eprec*(shrunk[1,0,1] + shrunk[0,1,1] + shrunk[1,1,0])
	val += dprec*shrunk[1,1,1]
	val += lprec*obs[0,0,0]
	shrunk[0,0,0] = val/prec
	diffs[0,0,0] = shrunk[0,0,0] - diffs[0,0,0]

	# top left corner on the upper face is at voxel 0,0,P-1
	val = sprec*(shrunk[1,0,P-1] + shrunk[0,1,P-1] + shrunk[0,0,P-2])
	val += eprec*(shrunk[1,0,P-2] + shrunk[0,1,P-2] + shrunk[1,1,P-1])
	val += dprec*shrunk[1,1,P-2]
	val += lprec*obs[0,0,P-1]
	shrunk[0,0,P-1] = val/prec
	diffs[0,0,P-1] = shrunk[0,0,P-1] - diffs[0,0,P-1]

	#------------------------------------------------

	# top right corner on the lower face is at voxel 0,N-1,0
	val = sprec*(shrunk[1,N-1,0] + shrunk[0,N-2,0] + shrunk[0,N-1,1])
	val += eprec*(shrunk[1,N-1,1] + shrunk[0,N-2,1] + shrunk[1,N-2,0])
	val += dprec*shrunk[1,N-2,1]
	val += lprec*obs[0,N-1,0]
	shrunk[0,N-1,0] = val/prec
	diffs[0,N-1,0] = shrunk[0,N-1,0] - diffs[0,N-1,0]

	# top right corner on the upper face is at voxel 0,N-1,P-1
	val = sprec*(shrunk[1,N-1,P-1] + shrunk[0,N-2,P-1] + shrunk[0,N-1,P-2])
	val += eprec*(shrunk[1,N-1,P-2] + shrunk[0,N-2,P-2] + shrunk[1,N-2,P-1])
	val += dprec*shrunk[1,N-2,P-2]
	val += lprec*obs[0,N-1,P-1]
	shrunk[0,N-1,P-1] = val/prec
	diffs[0,N-1,P-1] = shrunk[0,N-1,P-1] - diffs[0,N-1,P-1]

	#------------------------------------------------

	# bottom left corner on the lower face is at voxel M-1,0,0
	val = sprec*(shrunk[M-2,0,0] + shrunk[M-1,1,0] + shrunk[M-1,0,1])
	val += eprec*(shrunk[M-2,0,1] + shrunk[M-1,1,1] + shrunk[M-2,1,0])
	val += dprec*shrunk[M-2,1,1]
	val += lprec*obs[M-1,0,0]
	shrunk[M-1,0,0] = val/prec
	diffs[M-1,0,0] = shrunk[M-1,0,0] - diffs[M-1,0,0]

	# bottom left corner on the upper face is at voxel M-1,0,P-1
	val = sprec*(shrunk[M-2,0,P-1] + shrunk[M-1,1,P-1] + shrunk[M-1,0,P-2])
	val += eprec*(shrunk[M-2,0,P-2] + shrunk[M-1,1,P-2] + shrunk[M-2,1,P-1])
	val += dprec*shrunk[M-2,1,P-2]
	val += lprec*obs[M-1,0,P-1]
	shrunk[M-1,0,P-1] = val/prec
	diffs[M-1,0,P-1] = shrunk[M-1,0,P-1] - diffs[M-1,0,P-1]

	#------------------------------------------------

	# bottom right corner on the lower face is at voxel M-1,N-1,0
	val = sprec*(shrunk[M-2,N-1,0] + shrunk[M-1,N-2,0] + shrunk[M-1,N-1,1])
	val += eprec*(shrunk[M-2,N-1,1] + shrunk[M-1,N-2,1] + shrunk[M-2,N-2,0])
	val += dprec*shrunk[M-2,N-2,1]
	val += lprec*obs[M-1,N-1,0]
	shrunk[M-1,N-1,0] = val/prec
	diffs[M-1,N-1,0] = shrunk[M-1,N-1,0] - diffs[M-1,N-1,0]

	# bottom right corner on the upper face is at voxel M-1,N-1,P-1
	val = sprec*(shrunk[M-2,N-1,P-1] + shrunk[M-1,N-2,P-1]
					+ shrunk[M-1,N-1,P-2])
	val += eprec*(shrunk[M-2,N-1,P-2] + shrunk[M-1,N-2,P-2]
					+ shrunk[M-2,N-2,P-1])
	val += dprec*shrunk[M-2,N-2,P-2]
	val += lprec*obs[M-1,N-1,P-1]
	shrunk[M-1,N-1,P-1] = val/prec
	diffs[M-1,N-1,P-1] = shrunk[M-1,N-1,P-1] - diffs[M-1,N-1,P-1]


	#------------------------------------------------

	# edges
	prec = 4.0*sprec + 5.0*eprec +2.0*dprec + lprec


	# left side edge along the bottom face
	for idx in range(1,3):
		for i in prange(idx,M-1,2,nogil=True):
			val = sprec*(shrunk[i-1,0,0] + shrunk[i+1,0,0])
			val = val + sprec*(shrunk[i,1,0] + shrunk[i,0,1])

			val = val + eprec*(shrunk[i-1,0,1] + shrunk[i+1,0,1])
			val = val + eprec*(shrunk[i-1,1,0] + shrunk[i+1,1,0])
			val = val + eprec*shrunk[i,1,1]

			val = val + dprec*(shrunk[i-1,1,1] + shrunk[i+1,1,1])
			val = val + lprec*obs[i,0,0]

			shrunk[i,0,0] = val/prec
			diffs[i,0,0] = shrunk[i,0,0] - diffs[i,0,0]

	# left side edge along the top face
	for idx in range(1,3):
		for i in prange(idx,M-1,2,nogil=True):
			val = sprec*(shrunk[i-1,0,P-1] + shrunk[i+1,0,P-1])
			val = val + sprec*(shrunk[i,1,P-1] + shrunk[i,0,P-2])

			val = val + eprec*(shrunk[i-1,0,P-2] + shrunk[i+1,0,P-2])
			val = val + eprec*(shrunk[i-1,1,P-1] + shrunk[i+1,1,P-1])
			val = val + eprec*shrunk[i,1,P-2]

			val = val + dprec*(shrunk[i-1,1,P-2] + shrunk[i+1,1,P-2])
			val = val + lprec*obs[i,0,P-1]

			shrunk[i,0,P-1] = val/prec
			diffs[i,0,P-1] = shrunk[i,0,P-1] - diffs[i,0,P-1]


	# right side edge along the bottom face
	for idx in range(1,3):
		for i in prange(idx,M-1,2,nogil=True):
			val = sprec*(shrunk[i-1,N-1,0] + shrunk[i+1,N-1,0])
			val = val + sprec*(shrunk[i,N-2,0] + shrunk[i,N-1,1])

			val = val + eprec*(shrunk[i-1,N-1,1] + shrunk[i+1,N-1,1])
			val = val + eprec*(shrunk[i-1,N-2,0] + shrunk[i+1,N-2,0])
			val = val + eprec*shrunk[i,N-2,1]

			val = val + dprec*(shrunk[i-1,N-2,1] + shrunk[i+1,N-2,1])
			val = val + lprec*obs[i,N-1,0]

			shrunk[i,N-1,0] = val/prec
			diffs[i,N-1,0] = shrunk[i,N-1,0] - diffs[i,N-1,0]

	# right side edge along the top face
	for idx in range(1,3):
		for i in prange(idx,M-1,2,nogil=True):
			val = sprec*(shrunk[i-1,N-1,P-1] + shrunk[i+1,N-1,P-1])
			val = val + sprec*(shrunk[i,N-2,P-1] + shrunk[i,N-1,P-2])

			val = val + eprec*(shrunk[i-1,N-1,P-2] + shrunk[i+1,N-1,P-2])
			val = val + eprec*(shrunk[i-1,N-2,P-1] + shrunk[i+1,N-2,P-1])
			val = val + eprec*shrunk[i,N-2,P-2]

			val = val + dprec*(shrunk[i-1,N-2,P-2] + shrunk[i+1,N-2,P-2])
			val = val + lprec*obs[i,N-1,P-1]

			shrunk[i,N-1,P-1] = val/prec
			diffs[i,N-1,P-1] = shrunk[i,N-1,P-1] - diffs[i,N-1,P-1]


	# top side edge along the bottom face
	for idx in range(1,3):
		for i in prange(idx,N-1,2,nogil=True):
			val = sprec*(shrunk[0,i-1,0] + shrunk[0,i+1,0])
			val = val + sprec*(shrunk[1,i,0] + shrunk[0,i,1])

			val = val + eprec*(shrunk[0,i-1,1] + shrunk[0,i+1,1])
			val = val + eprec*(shrunk[1,i-1,0] + shrunk[1,i+1,0])
			val = val + eprec*shrunk[1,i,1]

			val = val + dprec*(shrunk[1,i-1,1] + shrunk[1,i+1,1])
			val = val + lprec*obs[0,i,0]

			shrunk[0,i,0] = val/prec
			diffs[0,i,0] = shrunk[0,i,0] - diffs[0,i,0]


	# top side edge along the top face
	for idx in range(1,3):
		for i in prange(idx,N-1,2,nogil=True):
			val = sprec*(shrunk[0,i-1,P-1] + shrunk[0,i+1,P-1])
			val = val + sprec*(shrunk[1,i,P-1] + shrunk[0,i,P-2])

			val = val + eprec*(shrunk[0,i-1,P-2] + shrunk[0,i+1,P-2])
			val = val + eprec*(shrunk[1,i-1,P-1] + shrunk[1,i+1,P-1])
			val = val + eprec*shrunk[1,i,P-2]

			val = val + dprec*(shrunk[1,i-1,P-2] + shrunk[1,i+1,P-2])
			val = val + lprec*obs[0,i,P-1]

			shrunk[0,i,P-1] = val/prec
			diffs[0,i,P-1] = shrunk[0,i,P-1] - diffs[0,i,P-1]


	# bottom side edge along the bottom face
	for idx in range(1,3):
		for i in prange(idx,N-1,2,nogil=True):
			val = sprec*(shrunk[M-1,i-1,0] + shrunk[M-1,i+1,0])
			val = val + sprec*(shrunk[M-2,i,0] + shrunk[M-1,i,1])

			val = val + eprec*(shrunk[M-1,i-1,1] + shrunk[M-1,i+1,1])
			val = val + eprec*(shrunk[M-2,i-1,0] + shrunk[M-2,i+1,0])
			val = val + eprec*shrunk[M-2,i,1]

			val = val + dprec*(shrunk[M-2,i-1,1] + shrunk[M-2,i+1,1])
			val = val + lprec*obs[M-1,i,0]

			shrunk[M-1,i,0] = val/prec
			diffs[M-1,i,0] = shrunk[M-1,i,0] - diffs[M-1,i,0]


	# bottom side edge along the top face
	for idx in range(1,3):
		for i in prange(idx,N-1,2,nogil=True):
			val = sprec*(shrunk[M-1,i-1,P-1] + shrunk[M-1,i+1,P-1])
			val = val + sprec*(shrunk[M-2,i,P-1] + shrunk[M-1,i,P-2])

			val = val + eprec*(shrunk[M-1,i-1,P-2] + shrunk[M-1,i+1,P-2])
			val = val + eprec*(shrunk[M-2,i-1,P-1] + shrunk[M-2,i+1,P-1])
			val = val + eprec*shrunk[M-2,i,P-2]

			val = val + dprec*(shrunk[M-2,i-1,P-2] + shrunk[M-2,i+1,P-2])
			val = val + lprec*obs[M-1,i,P-1]

			shrunk[M-1,i,P-1] = val/prec
			diffs[M-1,i,P-1] = shrunk[M-1,i,P-1] - diffs[M-1,i,P-1]


	# front side edge along the left face (front means at origin)
	for idx in range(1,3):
		for i in prange(idx,P-1,2,nogil=True):
			val = sprec*(shrunk[0,0,i-1] + shrunk[0,0,i+1])
			val = val + sprec*(shrunk[1,0,i] + shrunk[0,1,i])

			val = val + eprec*(shrunk[0,1,i-1] + shrunk[0,1,i+1])
			val = val + eprec*(shrunk[1,0,i-1] + shrunk[1,0,i+1])
			val = val + eprec*shrunk[1,1,i]

			val = val + dprec*(shrunk[1,1,i-1] + shrunk[1,1,i+1])
			val = val + lprec*obs[0,0,i]

			shrunk[0,0,i] = val/prec
			diffs[0,0,i] = shrunk[0,0,i] - diffs[0,0,i]


	# front side edge along the right face (front means at origin)
	for idx in range(1,3):
		for i in prange(idx,P-1,2,nogil=True):
			val = sprec*(shrunk[0,N-1,i-1] + shrunk[0,N-1,i+1])
			val = val + sprec*(shrunk[1,N-1,i] + shrunk[0,N-2,i])

			val = val + eprec*(shrunk[0,N-2,i-1] + shrunk[0,N-2,i+1])
			val = val + eprec*(shrunk[1,N-1,i-1] + shrunk[1,N-1,i+1])
			val = val + eprec*shrunk[1,N-2,i]

			val = val + dprec*(shrunk[1,N-2,i-1] + shrunk[1,N-2,i+1])
			val = val + lprec*obs[0,N-1,i]

			shrunk[0,N-1,i] = val/prec
			diffs[0,N-1,i] = shrunk[0,N-1,i] - diffs[0,N-1,i]



	# back side edge along the left face (back means away from origin)
	for idx in range(1,3):
		for i in prange(idx,P-1,2,nogil=True):
			val = sprec*(shrunk[M-1,0,i-1] + shrunk[M-1,0,i+1])
			val = val + sprec*(shrunk[M-2,0,i] + shrunk[M-1,1,i])

			val = val + eprec*(shrunk[M-1,1,i-1] + shrunk[M-1,1,i+1])
			val = val + eprec*(shrunk[M-2,0,i-1] + shrunk[M-2,0,i+1])
			val = val + eprec*shrunk[M-2,1,i]

			val = val + dprec*(shrunk[M-2,1,i-1] + shrunk[M-2,1,i+1])
			val = val + lprec*obs[M-1,0,i]

			shrunk[M-1,0,i] = val/prec
			diffs[M-1,0,i] = shrunk[M-1,0,i] - diffs[M-1,0,i]


	# back side edge along the right face (back means away from origin)
	for idx in range(1,3):
		for i in prange(idx,P-1,2,nogil=True):
			val = sprec*(shrunk[M-1,N-1,i-1] + shrunk[M-1,N-1,i+1])
			val = val + sprec*(shrunk[M-2,N-1,i] + shrunk[M-1,N-2,i])

			val = val + eprec*(shrunk[M-1,N-2,i-1] + shrunk[M-1,N-2,i+1])
			val = val + eprec*(shrunk[M-2,N-1,i-1] + shrunk[M-2,N-1,i+1])
			val = val + eprec*shrunk[M-2,N-2,i]

			val = val + dprec*(shrunk[M-2,N-2,i-1] + shrunk[M-2,N-2,i+1])
			val = val + lprec*obs[M-1,N-1,i]

			shrunk[M-1,N-1,i] = val/prec
			diffs[M-1,N-1,i] = shrunk[M-1,N-1,i] - diffs[M-1,N-1,i]


	#------------------------------------------------

	prec = 5.0*sprec + 8.0*eprec + 4.0*dprec + lprec


	# bottom face
	for idx in range(1,3):
		for i in prange(idx,M-1,2,nogil=True):
			for j in range(1,N-1):
				val = sprec*(shrunk[i-1,j,0] + shrunk[i+1,j,0])
				val = val + sprec*(shrunk[i,j-1,0] + shrunk[i,j+1,0])
				val = val + sprec*shrunk[i,j,1]

				val = val + eprec*(shrunk[i-1,j-1,0] + shrunk[i+1,j-1,0])
				val = val + eprec*(shrunk[i-1,j+1,0] + shrunk[i+1,j+1,0])
				val = val + eprec*(shrunk[i-1,j,1] + shrunk[i+1,j,1])
				val = val + eprec*(shrunk[i,j-1,1] + shrunk[i,j+1,1])

				val = val + dprec*(shrunk[i-1,j-1,1] + shrunk[i+1,j-1,1])
				val = val + dprec*(shrunk[i-1,j+1,1] + shrunk[i+1,j+1,1])

				val = val + lprec*obs[i,j,0]

				shrunk[i,j,0] = val/prec
				diffs[i,j,0] = shrunk[i,j,0] - diffs[i,j,0]


	# top face
	for idx in range(1,3):
		for i in prange(idx,M-1,2,nogil=True):
			for j in range(1,N-1):
				val = sprec*(shrunk[i-1,j,P-1] + shrunk[i+1,j,P-1])
				val = val + sprec*(shrunk[i,j-1,P-1] + shrunk[i,j+1,P-1])
				val = val + sprec*shrunk[i,j,P-2]

				val = val + eprec*(shrunk[i-1,j-1,P-1] + shrunk[i+1,j-1,P-1])
				val = val + eprec*(shrunk[i-1,j+1,P-1] + shrunk[i+1,j+1,P-1])
				val = val + eprec*(shrunk[i-1,j,P-2] + shrunk[i+1,j,P-2])
				val = val + eprec*(shrunk[i,j-1,P-2] + shrunk[i,j+1,P-2])

				val = val + dprec*(shrunk[i-1,j-1,P-2] + shrunk[i+1,j-1,P-2])
				val = val + dprec*(shrunk[i-1,j+1,P-2] + shrunk[i+1,j+1,P-2])

				val = val + lprec*obs[i,j,P-1]

				shrunk[i,j,P-1] = val/prec
				diffs[i,j,P-1] = shrunk[i,j,P-1] - diffs[i,j,P-1]


	# left face
	for idx in range(1,3):
		for i in prange(idx,M-1,2,nogil=True):
			for k in range(1,P-1):
				val = sprec*(shrunk[i-1,0,k] + shrunk[i+1,0,k])
				val = val + sprec*(shrunk[i,0,k-1] + shrunk[i,0,k+1])
				val = val + sprec*shrunk[i,1,k]

				val = val + eprec*(shrunk[i-1,0,k-1] + shrunk[i+1,0,k-1])
				val = val + eprec*(shrunk[i-1,0,k+1] + shrunk[i+1,0,k+1])
				val = val + eprec*(shrunk[i-1,1,k] + shrunk[i+1,1,k])
				val = val + eprec*(shrunk[i,1,k-1] + shrunk[i,1,k+1])

				val = val + dprec*(shrunk[i-1,1,k-1] + shrunk[i+1,1,k-1])
				val = val + dprec*(shrunk[i-1,1,k+1] + shrunk[i+1,1,k+1])

				val = val + lprec*obs[i,0,k]

				shrunk[i,0,k] = val/prec
				diffs[i,0,k] = shrunk[i,0,k] - diffs[i,0,k]



	# right face
	for idx in range(1,3):
		for i in prange(idx,M-1,2,nogil=True):
			for k in range(1,P-1):
				val = sprec*(shrunk[i-1,N-1,k] + shrunk[i+1,N-1,k])
				val = val + sprec*(shrunk[i,N-1,k-1] + shrunk[i,N-1,k+1])
				val = val + sprec*shrunk[i,N-2,k]

				val = val + eprec*(shrunk[i-1,N-1,k-1] + shrunk[i+1,N-1,k-1])
				val = val + eprec*(shrunk[i-1,N-1,k+1] + shrunk[i+1,N-1,k+1])
				val = val + eprec*(shrunk[i-1,N-2,k] + shrunk[i+1,N-2,k])
				val = val + eprec*(shrunk[i,N-2,k-1] + shrunk[i,N-2,k+1])

				val = val + dprec*(shrunk[i-1,N-2,k-1] + shrunk[i+1,N-2,k-1])
				val = val + dprec*(shrunk[i-1,N-2,k+1] + shrunk[i+1,N-2,k+1])

				val = val + lprec*obs[i,N-1,k]

				shrunk[i,N-1,k] = val/prec
				diffs[i,N-1,k] = shrunk[i,N-1,k] - diffs[i,N-1,k]


	# back face
	for idx in range(1,3):
		for j in prange(idx,N-1,2,nogil=True):
			for k in range(1,P-1):
				val = sprec*(shrunk[0,j-1,k] + shrunk[0,j+1,k])
				val = val + sprec*(shrunk[0,j,k-1] + shrunk[0,j,k+1])
				val = val + sprec*shrunk[1,j,k]

				val = val + eprec*(shrunk[0,j-1,k-1] + shrunk[0,j+1,k-1])
				val = val + eprec*(shrunk[0,j-1,k+1] + shrunk[0,j+1,k+1])
				val = val + eprec*(shrunk[1,j-1,k] + shrunk[1,j+1,k])
				val = val + eprec*(shrunk[1,j,k-1] + shrunk[1,j,k+1])

				val = val + dprec*(shrunk[1,j-1,k-1] + shrunk[1,j+1,k-1])
				val = val + dprec*(shrunk[1,j-1,k+1] + shrunk[1,j+1,k+1])

				val = val + lprec*obs[0,j,k]

				shrunk[0,j,k] = val/prec
				diffs[0,j,k] = shrunk[0,j,k] - diffs[0,j,k]


	# front face
	for idx in range(1,3):
		for j in prange(idx,N-1,2,nogil=True):
			for k in range(1,P-1):
				val = sprec*(shrunk[M-1,j-1,k] + shrunk[M-1,j+1,k])
				val = val + sprec*(shrunk[M-1,j,k-1] + shrunk[M-1,j,k+1])
				val = val + sprec*shrunk[M-2,j,k]

				val = val + eprec*(shrunk[M-1,j-1,k-1] + shrunk[M-1,j+1,k-1])
				val = val + eprec*(shrunk[M-1,j-1,k+1] + shrunk[M-1,j+1,k+1])
				val = val + eprec*(shrunk[M-2,j-1,k] + shrunk[M-2,j+1,k])
				val = val + eprec*(shrunk[M-2,j,k-1] + shrunk[M-2,j,k+1])

				val = val + dprec*(shrunk[M-2,j-1,k-1] + shrunk[M-2,j+1,k-1])
				val = val + dprec*(shrunk[M-2,j-1,k+1] + shrunk[M-2,j+1,k+1])

				val = val + lprec*obs[M-1,j,k]

				shrunk[M-1,j,k] = val/prec

				diffs[M-1,j,k] = shrunk[M-1,j,k] - diffs[M-1,j,k]


	#------------------------------------------------

	# middle
	prec = 6.0*sprec + 12.0*eprec + 8.0*dprec
	prec = prec + lprec

	for idx in range(1,3):
		for i in prange(idx,M-1,2,nogil=True):
			for j in range(1,N-1):
				for k in range(1,P-1):
					val = sprec*(shrunk[i,j,k-1] + shrunk[i,j,k+1])
					val = val + sprec*(shrunk[i,j-1,k] + shrunk[i,j+1,k])
					val = val + sprec*(shrunk[i-1,j,k] + shrunk[i+1,j,k])

					val = val + eprec*(shrunk[i,j-1,k-1] + shrunk[i,j+1,k-1])
					val = val + eprec*(shrunk[i-1,j,k-1] + shrunk[i+1,j,k-1])

					val = val + eprec*(shrunk[i-1,j-1,k] + shrunk[i-1,j+1,k])
					val = val + eprec*(shrunk[i+1,j-1,k] + shrunk[i+1,j+1,k])

					val = val + eprec*(shrunk[i,j-1,k+1] + shrunk[i,j+1,k+1])
					val = val + eprec*(shrunk[i-1,j,k+1] + shrunk[i+1,j,k+1])

					val = val + dprec*(shrunk[i-1,j-1,k-1]+shrunk[i-1,j+1,k-1])
					val = val + dprec*(shrunk[i+1,j-1,k-1]+shrunk[i+1,j+1,k-1])
					val = val + dprec*(shrunk[i-1,j-1,k+1]+shrunk[i-1,j+1,k+1])
					val = val + dprec*(shrunk[i+1,j-1,k+1]+shrunk[i+1,j+1,k+1])

					val = val + lprec*obs[i,j,k]

					shrunk[i,j,k] = val/prec

					diffs[i,j,k] = shrunk[i,j,k] - diffs[i,j,k]

	if not relax:
		return np.abs(diffs).max()

	for idx in range(1,3):
		for i in prange(idx,M-1,2,nogil=True):
			for j in range(N):
				for k in range(P):
					shrunk[i,j,k] += relax_coeff*diffs[i,j,k]

	return np.abs(diffs).max()

