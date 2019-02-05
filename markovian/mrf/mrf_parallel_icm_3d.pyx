import numpy as np
cimport numpy as cnp
import cython
from cython.parallel import prange, parallel

# should be cimport to run in parallel
#from libc.math import exp as exp_c
from libc.math cimport fabs as fabs_c

from redescendl import redescend_normal1, redescend_normal2


# This isn't parallel because it doesn't condition on sets from the
# global Markov property as easily as do 2D and 3D arrays.  The number
# of processors/threads will need to be identified ahead of time to
# condition on end points and then mid-points.
@cython.boundscheck(False)
@cython.cdivision(True)
def shrink_mrf1_icm(double[:] vec,
			double pprec, double lprec, double converged=1e-6):

	cdef double[:] vec1 = vec.copy()
	cdef double[:] old_vec = vec.copy()
	cdef double[:] diffs = np.zeros_like(vec)

	cdef int N = vec.shape[0]

	cdef int i
	cdef start_idx
	cdef double val

	cdef double prec1 = lprec + fabs_c(pprec)
	cdef double prec2 = lprec + 2.0*fabs_c(pprec)

	while 1:
		old_vec = vec1.copy()

		vec1[0] = (pprec*vec1[1] + lprec*vec[0])/prec1
		diffs[0] = vec1[0] - old_vec[0]

		for i in range(1,N-1):
			val = pprec*(vec1[i-1] + vec1[i+1]) + lprec*vec[i]
			vec1[i] = val/prec2
			diffs[i] = vec1[i] - old_vec[i]

		vec1[N-1] = (pprec*vec1[N-2] + lprec*vec[N-1])/prec1
		diffs[N-1] = vec1[N-1] - old_vec[N-1]
		
		# convergence is quadratic the sanity check needs to be on how
		# small the value of converged is
		# if diffs.max() < converged: break
		if np.abs(diffs).max() < converged: break

	return np.asarray(vec1)

#-------------------------------------------------

def shrink_mrf1_redescend(double[:] vec,
			double pprec, double lprec, double cval, 
			int max_iter=30, double converged=1e-6):


	cdef int N = vec.shape[0]

	cdef double[:] vec1 = np.zeros_like(vec)
	cdef double[:] resids = np.zeros_like(vec)
	cdef double[:] redescended = np.zeros_like(vec)
	cdef double[:] diffs = np.zeros_like(vec)

	cdef double[:] shrunk = shrink_mrf1_icm(vec, pprec, lprec, converged)

	cdef double[:] shrunk_old = shrunk.copy()

	cdef int iter_count = 0
	cdef int i


	while 1:
		for i in range(N):
			resids[i] = vec[i] - shrunk[i]

		redescended = redescend_normal1(np.asarray(resids), cval)

		for i in range(N):
			vec1[i] = shrunk[i] + redescended[i]

		shrunk = shrink_mrf1_icm(vec1, pprec, lprec, converged)

		iter_count += 1
		if not iter_count < max_iter: break

		diff = 0.0
		for i in range(N):
			diffs[i] = shrunk[i] - shrunk_old[i]

		if np.abs(diffs).max() < converged: break

		# Seems to go well without relaxing convergence
		#if iter_count > 4:
		#	for i in range(N): shrunk[i] += 0.3*diffs[i]
				

		shrunk_old = shrunk.copy()


	return (np.asarray(shrunk),iter_count)

#-------------------------------------------------



@cython.boundscheck(False)
@cython.cdivision(True)
def shrink_mrf2_icm(double[:,:] observed,
			double prior_edge_prec, double prior_diag_prec,
			double likelihood_prec, double converged=1e-6):

	cdef int N = observed.shape[0]
	cdef int P = observed.shape[1]
	cdef double[:,:] shrunk = observed.copy()
	cdef double[:,:] old_shrunk = observed.copy()
	cdef double[:,:] diffs = np.zeros_like(observed)

	cdef int i, j
	cdef int start_idx

	cdef double prec, val
	
	# just to make the code a bit more compact
	cdef double eprec = prior_edge_prec
	cdef double dprec = prior_diag_prec
	cdef double lprec = likelihood_prec

	
	while 1:
		old_shrunk = shrunk.copy()

		# corners
		prec = 2.0*fabs_c(eprec) + fabs_c(dprec) + lprec

		# top left corner
		val = eprec*(shrunk[0,1] + shrunk[1,0]) + dprec*shrunk[1,1]
		val = val + lprec*observed[0,0]
		shrunk[0,0] = val/prec
		diffs[0,0] = shrunk[0,0] - old_shrunk[0,0]
		

		# top right corner
		val = eprec*(shrunk[0,P-2] + shrunk[1,P-1]) + dprec*shrunk[1,P-2]
		val = val + lprec*observed[0,P-1]
		shrunk[0,P-1] = val/prec
		diffs[0,P-1] = shrunk[0,P-1] - old_shrunk[0,P-1]

		# bottom left corner
		val = eprec*(shrunk[N-2,0] + shrunk[N-1,1]) + dprec*shrunk[N-2,1]
		val = val + lprec*observed[N-1,0]
		shrunk[N-1,0] = val/prec
		diffs[N-1,0] = shrunk[N-1,0] - old_shrunk[N-1,0]

		# bottom right corner
		val = eprec*(shrunk[N-2,P-1] + shrunk[N-1,P-2]) + dprec*shrunk[N-2,P-2]
		val = val + lprec*observed[N-1,P-1]
		shrunk[N-1,P-1] = val/prec
		diffs[N-1,P-1] = shrunk[N-1,P-1] - old_shrunk[N-1,P-1]


		# edges
		prec = 3.0*fabs_c(eprec) + 2.0*fabs_c(dprec) + lprec

		# across top then bottom
		for j in range(1,P-1):
			val = eprec*(shrunk[0,j-1] + shrunk[0,j+1] + shrunk[1,j])
			val = val + dprec*(shrunk[1,j-1] + shrunk[1,j+1])
			val = val + lprec*observed[0,j]
			shrunk[0,j] = val/prec
			diffs[0,j] = shrunk[0,j] - old_shrunk[0,j]

			val = eprec*(shrunk[N-1,j-1] + shrunk[N-1,j+1] + shrunk[N-2,j])
			val = val + dprec*(shrunk[N-2,j-1] + shrunk[N-2,j+1])
			val = val + lprec*observed[N-1,j]
			shrunk[N-1,j] = val/prec
			diffs[N-1,j] = shrunk[N-1,j] - old_shrunk[N-1,j]


		# down left then right
		for i in range(1,N-1):
			val = eprec*(shrunk[i-1,0] + shrunk[i+1,0] + shrunk[i,1])
			val = val + dprec*(shrunk[i-1,1] + shrunk[i+1,1])
			val = val + lprec*observed[i,0]
			shrunk[i,0] = val/prec
			diffs[i,0] = shrunk[i,0] - old_shrunk[i,0]

			val = eprec*(shrunk[i-1,P-1] + shrunk[i+1,P-1] + shrunk[i,P-2])
			val = val + dprec*(shrunk[i-1,P-2] + shrunk[i+1,P-2])
			val = val + lprec*observed[i,P-1]
			shrunk[i,P-1] = val/prec
			diffs[i,P-1] = shrunk[i,P-1] - old_shrunk[i,P-1]


		# middle precision
		prec = 4.0*fabs_c(eprec) + 4.0*fabs_c(dprec) + lprec

		for start_idx in range(1,3):
			for i in prange(start_idx,N-1,2,nogil=True):
				for j in range(1,P-1):
					val = eprec*(shrunk[i-1,j] + shrunk[i+1,j])
					val = val + eprec*(shrunk[i,j-1] + shrunk[i,j+1])
					val = val + dprec*(shrunk[i-1,j-1] + shrunk[i-1,j+1])
					val = val + dprec*(shrunk[i+1,j-1] + shrunk[i+1,j+1])
					val = val + lprec*observed[i,j]
					shrunk[i,j] = val/prec
					diffs[i,j] = shrunk[i,j] - old_shrunk[i,j]

		
		if np.abs(diffs).max() < converged: break

	return np.asarray(shrunk)


#-------------------------------------------------------------


def shrink_mrf2_redescend(double [:,:] obs,
			double edge_prec, double diag_prec, double lhood_prec,
			double cval, int max_iter=30, double converged=1e-6):

	cdef int N1 = obs.shape[0]
	cdef int N2 = obs.shape[1]

	cdef double [:,:] resids = np.zeros_like(obs)
	cdef double [:,:] shrunk = np.zeros_like(obs)
	cdef double [:,:] redescended = np.zeros_like(obs)
	cdef double [:,:] diff_arr = np.zeros_like(obs)

	shrunk = shrink_mrf2_icm(obs,edge_prec,diag_prec,lhood_prec,converged)

	cdef double[:,:] shrunk_old = shrunk.copy()

	cdef int iter_count = 0

	cdef int i,j

	cdef double diff

	while 1:
		for i in range(N1):
			for j in range(N2):
				resids[i,j] = obs[i,j] - shrunk[i,j]

		redescended = redescend_normal2(np.asarray(resids), cval)

		for i in range(N1):
			for j in range(N2):
				redescended[i,j] += shrunk[i,j]

		shrunk = shrink_mrf2_icm(redescended,edge_prec,diag_prec,
					lhood_prec, converged)

		iter_count += 1

		if not iter_count < max_iter: break

		for i in range(N1):
			for j in range(N2):
				diff_arr[i,j] = shrunk[i,j] - shrunk_old[i,j]

		diff = np.abs(diff_arr).max()
		if diff < converged: break

#		skip relaxing the solution for now
#		if iter_count > 3: shrunk += 0.35*(shrunk - shrunk_old)

		shrunk_old = shrunk.copy()

	return (np.asarray(shrunk),iter_count)

#--------------------------------------------------------------------



# forget why this is here 2019-01-25
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline double f(double x) nogil:
	if x > 0.5:
		return fabs_c(x)
	else:
		return 0


#herehere
# used by a thread to run an iteration of icm
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline int shrink_mrf3_icm_thread(
			double[:,:,:] obs, double[:,:,:] shrunk,
			double[:,:,:] diffs, int idx,
			double sprec, double eprec, double dprec, double lprec,
			double converged=1e-6) nogil:

	cdef double val, prec
	cdef int j, k

	cdef int N = obs.shape[1]
	cdef int P = obs.shape[2]

	prec = 6.0*fabs_c(sprec) + 12.0*fabs_c(eprec) + 8.0*fabs_c(dprec)
	prec = prec + lprec

	with parallel():
		for j in range(1,N-1):
			for k in range(1,P-1):
				val = sprec*(shrunk[idx,j,k-1] + shrunk[idx,j,k+1])
				val = val + sprec*(shrunk[idx,j-1,k] + shrunk[idx,j+1,k])
				val = val + sprec*(shrunk[idx-1,j,k] + shrunk[idx+1,j,k])

				val = val + eprec*(shrunk[idx,j-1,k-1] + shrunk[idx,j+1,k-1])
				val = val + eprec*(shrunk[idx-1,j,k-1] + shrunk[idx+1,j,k-1])

				val = val + eprec*(shrunk[idx-1,j-1,k] + shrunk[idx-1,j+1,k])
				val = val + eprec*(shrunk[idx+1,j-1,k] + shrunk[idx+1,j+1,k])

				val = val + eprec*(shrunk[idx,j-1,k+1] + shrunk[idx,j+1,k+1])
				val = val + eprec*(shrunk[idx-1,j,k+1] + shrunk[idx+1,j,k+1])

				val = val + dprec*(shrunk[idx-1,j-1,k-1]+shrunk[idx-1,j+1,k-1])
				val = val + dprec*(shrunk[idx+1,j-1,k-1]+shrunk[idx+1,j+1,k-1])
				val = val + dprec*(shrunk[idx-1,j-1,k+1]+shrunk[idx-1,j+1,k+1])
				val = val + dprec*(shrunk[idx+1,j-1,k+1]+shrunk[idx+1,j+1,k+1])

				val = val + lprec*obs[idx,j,k]

				shrunk[idx,j,k] = val/prec

				diffs[idx,j,k] = fabs_c(shrunk[idx,j,k] - diffs[idx,j,k])

	return 0



#@cython.boundscheck(False)
#@cython.cdivision(True)
def shrink_mrf3_icm(double[:,:,:] obs,
			double prior_side_prec, double prior_edge_prec,
			double prior_diag_prec, double likelihood_prec,
			double converged=1e-6):

	cdef int M = obs.shape[0]
	cdef int N = obs.shape[1]
	cdef int P = obs.shape[2]
	cdef double[:,:,:] shrunk = obs.copy()
	cdef double[:,:,:] diffs = obs.copy()

	cdef int niter = 0
	cdef int i, j, k
	cdef double prec, val
	
	# just to make the code a bit more compact
	cdef double sprec = prior_side_prec
	cdef double eprec = prior_edge_prec
	cdef double dprec = prior_diag_prec
	cdef double lprec = likelihood_prec

	
	while 1:
		# this also doubles to keep trackof the absolute difference
		diffs = shrunk.copy()

		# corners
		prec = 3.0*sprec + 3.0*eprec + dprec + lprec

		# top left corner on the lower face is at voxel 0,0,0
		val = sprec*(shrunk[1,0,0] + shrunk[0,1,0] + shrunk[0,0,1])
		val = val + eprec*(shrunk[1,0,1] + shrunk[0,1,1] + shrunk[1,1,0])
		val = val + dprec*shrunk[1,1,1]
		val = val + lprec*obs[0,0,0]
		shrunk[0,0,0] = val/prec
		diffs[0,0,0] = shrunk[0,0,0] - diffs[0,0,0]

		# top left corner on the upper face is at voxel 0,0,P-1
		val = sprec*(shrunk[1,0,P-1] + shrunk[0,1,P-1] + shrunk[0,0,P-2])
		val = val + eprec*(shrunk[1,0,P-2] + shrunk[0,1,P-2] + shrunk[1,1,P-1])
		val = val + dprec*shrunk[1,1,P-2]
		val = val + lprec*obs[0,0,P-1]
		shrunk[0,0,P-1] = val/prec
		diffs[0,0,P-1] = shrunk[0,0,P-1] - diffs[0,0,P-1]

		#------------------------------------------------

		# top right corner on the lower face is at voxel 0,N-1,0
		val = sprec*(shrunk[1,N-1,0] + shrunk[0,N-2,0] + shrunk[0,N-1,1])
		val = val + eprec*(shrunk[1,N-1,1] + shrunk[0,N-2,1] + shrunk[1,N-2,0])
		val = val + dprec*shrunk[1,N-2,1]
		val = val + lprec*obs[0,N-1,0]
		shrunk[0,N-1,0] = val/prec
		diffs[0,N-1,0] = shrunk[0,N-1,0] - diffs[0,N-1,0]

		# top right corner on the upper face is at voxel 0,N-1,P-1
		val = sprec*(shrunk[1,N-1,P-1] + shrunk[0,N-2,P-1] + shrunk[0,N-1,P-2])
		val = val + eprec*(shrunk[1,N-1,P-2] + shrunk[0,N-2,P-2] +
						shrunk[1,N-2,P-1])
		val = val + dprec*shrunk[1,N-2,P-2]
		val = val + lprec*obs[0,N-1,P-1]
		shrunk[0,N-1,P-1] = val/prec
		diffs[0,N-1,P-1] = shrunk[0,N-1,P-1] - diffs[0,N-1,P-1]

		#------------------------------------------------

		# bottom left corner on the lower face is at voxel M-1,0,0
		val = sprec*(shrunk[M-2,0,0] + shrunk[M-1,1,0] + shrunk[M-1,0,1])
		val = val + eprec*(shrunk[M-2,0,1] + shrunk[M-1,1,1] + shrunk[M-2,1,0])
		val = val + dprec*shrunk[M-2,1,1]
		val = val + lprec*obs[M-1,0,0]
		shrunk[M-1,0,0] = val/prec
		diffs[M-1,0,0] = shrunk[M-1,0,0] - diffs[M-1,0,0]

		# bottom left corner on the upper face is at voxel M-1,0,P-1
		val = sprec*(shrunk[M-2,0,P-1] + shrunk[M-1,1,P-1] + shrunk[M-1,0,P-2])
		val = val + eprec*(shrunk[M-2,0,P-2] + shrunk[M-1,1,P-2] +
						shrunk[M-2,1,P-1])
		val = val + dprec*shrunk[M-2,1,P-2]
		val = val + lprec*obs[M-1,0,P-1]
		shrunk[M-1,0,P-1] = val/prec
		diffs[M-1,0,P-1] = shrunk[M-1,0,P-1] - diffs[M-1,0,P-1]


		# bottom right corner on the lower face is at voxel M-1,N-1,0
		val = sprec*(shrunk[M-2,N-1,0] + shrunk[M-1,N-2,0] + shrunk[M-1,N-1,1])
		val = val + eprec*(shrunk[M-2,N-1,1] + shrunk[M-1,N-2,1]
					+ shrunk[M-2,N-2,0])
		val = val + dprec*shrunk[M-2,N-2,1]
		val = val + lprec*obs[M-1,N-1,0]
		shrunk[M-1,N-1,0] = val/prec
		diffs[M-1,N-1,0] = shrunk[M-1,N-1,0] - diffs[M-1,N-1,0]

		# bottom right corner on the upper face is at voxel M-1,N-1,P-1
		val = sprec*(shrunk[M-2,N-1,P-1] + shrunk[M-1,N-2,P-1]
						+ shrunk[M-1,N-1,P-2])
		val = val + eprec*(shrunk[M-2,N-1,P-2] + shrunk[M-1,N-2,P-2]
						+ shrunk[M-2,N-2,P-1])
		val = val + dprec*shrunk[M-2,N-2,P-2]
		val = val + lprec*obs[M-1,N-1,P-1]
		shrunk[M-1,N-1,P-1] = val/prec
		diffs[M-1,N-1,P-1] = shrunk[M-1,N-1,P-1] - diffs[M-1,N-1,P-1]


		#------------------------------------------------

		# edges
		prec = 4.0*sprec + 5.0*eprec +2.0*dprec + lprec


		# left side edge along the bottom face
		for i in range(1,M-1):
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
		for i in range(1,M-1):
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
		for i in range(1,M-1):
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
		for i in range(1,M-1):
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
		for i in range(1,N-1):
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
		for i in range(1,N-1):
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
		for i in range(1,N-1):
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
		for i in range(1,N-1):
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
		for i in range(1,P-1):
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
		for i in range(1,P-1):
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
		for i in range(1,P-1):
			val = sprec*(shrunk[M-1,0,i-1] + shrunk[M-1,0,i+1])
			val = val + sprec*(shrunk[M-2,0,i] + shrunk[M-1,1,i])

			val = val + eprec*(shrunk[M-1,1,i-1] + shrunk[M-1,1,i+1])
			val = val + eprec*(shrunk[M-2,0,i-1] + shrunk[M-2,0,i+1])
			val = val + eprec*shrunk[M-2,1,i]

			val = val + dprec*(shrunk[M-2,1,i-1] + shrunk[M-2,1,i+1])
			val = val + lprec*obs[M-1,0,i]

			shrunk[M-1,0,i] = val/prec
			diffs[M-1,0,i] = shrunk[M-1,0,i] - diffs[M-1,0,i]


#		# back side edge along the right face (back means away from origin)
		for i in range(1,P-1):
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
		for i in range(1,M-1):
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
		for i in range(1,M-1):
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
		for i in range(1,M-1):
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
		for i in range(1,M-1):
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
				diffs[i,N-1,k] =shrunk[i,N-1,k] - diffs[i,N-1,k]


		# back face
		for j in range(1,N-1):
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
		for j in range(1,N-1):
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


		for i in range(1,M-1):
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

		niter += 1
		print('max diff =', np.abs(diffs).max())
		if np.abs(diffs).max() < converged: break

		if niter > 3:
			for i in range(M):
				for j in range(N):
					for k in range(P):
						shrunk[i,j,k] += 0.35*diffs[i,j,k]

	return np.asarray(shrunk)



@cython.boundscheck(False)
@cython.cdivision(True)
def shrink_mrf3_icm_iter(double[:,:,:] obs, double[:,:,:] shrunk,
			double prior_side_prec, double prior_edge_prec,
			double prior_diag_prec, double likelihood_prec,
			double converged=1e-6):

	cdef int M = obs.shape[0]
	cdef int N = obs.shape[1]
	cdef int P = obs.shape[2]
	cdef double[:,:,:] diffs = shrunk.copy()

#	cdef int niter = 0
	cdef int i, j, k
	cdef double prec, val
	
	# just to make the code a bit more compact
	cdef double sprec = prior_side_prec
	cdef double eprec = prior_edge_prec
	cdef double dprec = prior_diag_prec
	cdef double lprec = likelihood_prec

	
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
	for i in range(1,M-1):
		val = sprec*(shrunk[i-1,0,0] + shrunk[i+1,0,0])
		val += sprec*(shrunk[i,1,0] + shrunk[i,0,1])

		val += eprec*(shrunk[i-1,0,1] + shrunk[i+1,0,1])
		val += eprec*(shrunk[i-1,1,0] + shrunk[i+1,1,0])
		val += eprec*shrunk[i,1,1]

		val += dprec*(shrunk[i-1,1,1] + shrunk[i+1,1,1])
		val += lprec*obs[i,0,0]

		shrunk[i,0,0] = val/prec
		diffs[i,0,0] = shrunk[i,0,0] - diffs[i,0,0]

	# left side edge along the top face
	for i in range(1,M-1):
		val = sprec*(shrunk[i-1,0,P-1] + shrunk[i+1,0,P-1])
		val += sprec*(shrunk[i,1,P-1] + shrunk[i,0,P-2])

		val += eprec*(shrunk[i-1,0,P-2] + shrunk[i+1,0,P-2])
		val += eprec*(shrunk[i-1,1,P-1] + shrunk[i+1,1,P-1])
		val += eprec*shrunk[i,1,P-2]

		val += dprec*(shrunk[i-1,1,P-2] + shrunk[i+1,1,P-2])
		val += lprec*obs[i,0,P-1]

		shrunk[i,0,P-1] = val/prec
		diffs[i,0,P-1] = shrunk[i,0,P-1] - diffs[i,0,P-1]


	# right side edge along the bottom face
	for i in range(1,M-1):
		val = sprec*(shrunk[i-1,N-1,0] + shrunk[i+1,N-1,0])
		val += sprec*(shrunk[i,N-2,0] + shrunk[i,N-1,1])

		val += eprec*(shrunk[i-1,N-1,1] + shrunk[i+1,N-1,1])
		val += eprec*(shrunk[i-1,N-2,0] + shrunk[i+1,N-2,0])
		val += eprec*shrunk[i,N-2,1]

		val += dprec*(shrunk[i-1,N-2,1] + shrunk[i+1,N-2,1])
		val += lprec*obs[i,N-1,0]

		shrunk[i,N-1,0] = val/prec
		diffs[i,N-1,0] = shrunk[i,N-1,0] - diffs[i,N-1,0]

	# right side edge along the top face
	for i in range(1,M-1):
		val = sprec*(shrunk[i-1,N-1,P-1] + shrunk[i+1,N-1,P-1])
		val += sprec*(shrunk[i,N-2,P-1] + shrunk[i,N-1,P-2])

		val += eprec*(shrunk[i-1,N-1,P-2] + shrunk[i+1,N-1,P-2])
		val += eprec*(shrunk[i-1,N-2,P-1] + shrunk[i+1,N-2,P-1])
		val += eprec*shrunk[i,N-2,P-2]

		val += dprec*(shrunk[i-1,N-2,P-2] + shrunk[i+1,N-2,P-2])
		val += lprec*obs[i,N-1,P-1]

		shrunk[i,N-1,P-1] = val/prec
		diffs[i,N-1,P-1] = shrunk[i,N-1,P-1] - diffs[i,N-1,P-1]


	# top side edge along the bottom face
	for i in range(1,N-1):
		val = sprec*(shrunk[0,i-1,0] + shrunk[0,i+1,0])
		val += sprec*(shrunk[1,i,0] + shrunk[0,i,1])

		val += eprec*(shrunk[0,i-1,1] + shrunk[0,i+1,1])
		val += eprec*(shrunk[1,i-1,0] + shrunk[1,i+1,0])
		val += eprec*shrunk[1,i,1]

		val += dprec*(shrunk[1,i-1,1] + shrunk[1,i+1,1])
		val += lprec*obs[0,i,0]

		shrunk[0,i,0] = val/prec
		diffs[0,i,0] = shrunk[0,i,0] - diffs[0,i,0]


	# top side edge along the top face
	for i in range(1,N-1):
		val = sprec*(shrunk[0,i-1,P-1] + shrunk[0,i+1,P-1])
		val += sprec*(shrunk[1,i,P-1] + shrunk[0,i,P-2])

		val += eprec*(shrunk[0,i-1,P-2] + shrunk[0,i+1,P-2])
		val += eprec*(shrunk[1,i-1,P-1] + shrunk[1,i+1,P-1])
		val += eprec*shrunk[1,i,P-2]

		val += dprec*(shrunk[1,i-1,P-2] + shrunk[1,i+1,P-2])
		val += lprec*obs[0,i,P-1]

		shrunk[0,i,P-1] = val/prec
		diffs[0,i,P-1] = shrunk[0,i,P-1] - diffs[0,i,P-1]


	# bottom side edge along the bottom face
	for i in range(1,N-1):
		val = sprec*(shrunk[M-1,i-1,0] + shrunk[M-1,i+1,0])
		val += sprec*(shrunk[M-2,i,0] + shrunk[M-1,i,1])

		val += eprec*(shrunk[M-1,i-1,1] + shrunk[M-1,i+1,1])
		val += eprec*(shrunk[M-2,i-1,0] + shrunk[M-2,i+1,0])
		val += eprec*shrunk[M-2,i,1]

		val += dprec*(shrunk[M-2,i-1,1] + shrunk[M-2,i+1,1])
		val += lprec*obs[M-1,i,0]

		shrunk[M-1,i,0] = val/prec
		diffs[M-1,i,0] = shrunk[M-1,i,0] - diffs[M-1,i,0]


	# bottom side edge along the top face
	for i in range(1,N-1):
		val = sprec*(shrunk[M-1,i-1,P-1] + shrunk[M-1,i+1,P-1])
		val += sprec*(shrunk[M-2,i,P-1] + shrunk[M-1,i,P-2])

		val += eprec*(shrunk[M-1,i-1,P-2] + shrunk[M-1,i+1,P-2])
		val += eprec*(shrunk[M-2,i-1,P-1] + shrunk[M-2,i+1,P-1])
		val += eprec*shrunk[M-2,i,P-2]

		val += dprec*(shrunk[M-2,i-1,P-2] + shrunk[M-2,i+1,P-2])
		val += lprec*obs[M-1,i,P-1]

		shrunk[M-1,i,P-1] = val/prec
		diffs[M-1,i,P-1] = shrunk[M-1,i,P-1] - diffs[M-1,i,P-1]


	# front side edge along the left face (front means at origin)
	for i in range(1,P-1):
		val = sprec*(shrunk[0,0,i-1] + shrunk[0,0,i+1])
		val += sprec*(shrunk[1,0,i] + shrunk[0,1,i])

		val += eprec*(shrunk[0,1,i-1] + shrunk[0,1,i+1])
		val += eprec*(shrunk[1,0,i-1] + shrunk[1,0,i+1])
		val += eprec*shrunk[1,1,i]

		val += dprec*(shrunk[1,1,i-1] + shrunk[1,1,i+1])
		val += lprec*obs[0,0,i]

		shrunk[0,0,i] = val/prec
		diffs[0,0,i] = shrunk[0,0,i] - diffs[0,0,i]


	# front side edge along the right face (front means at origin)
	for i in range(1,P-1):
		val = sprec*(shrunk[0,N-1,i-1] + shrunk[0,N-1,i+1])
		val += sprec*(shrunk[1,N-1,i] + shrunk[0,N-2,i])

		val += eprec*(shrunk[0,N-2,i-1] + shrunk[0,N-2,i+1])
		val += eprec*(shrunk[1,N-1,i-1] + shrunk[1,N-1,i+1])
		val += eprec*shrunk[1,N-2,i]

		val += dprec*(shrunk[1,N-2,i-1] + shrunk[1,N-2,i+1])
		val += lprec*obs[0,N-1,i]

		shrunk[0,N-1,i] = val/prec
		diffs[0,N-1,i] = shrunk[0,N-1,i] - diffs[0,N-1,i]



	# back side edge along the left face (back means away from origin)
	for i in range(1,P-1):
		val = sprec*(shrunk[M-1,0,i-1] + shrunk[M-1,0,i+1])
		val += sprec*(shrunk[M-2,0,i] + shrunk[M-1,1,i])

		val += eprec*(shrunk[M-1,1,i-1] + shrunk[M-1,1,i+1])
		val += eprec*(shrunk[M-2,0,i-1] + shrunk[M-2,0,i+1])
		val += eprec*shrunk[M-2,1,i]

		val += dprec*(shrunk[M-2,1,i-1] + shrunk[M-2,1,i+1])
		val += lprec*obs[M-1,0,i]

		shrunk[M-1,0,i] = val/prec
		diffs[M-1,0,i] = shrunk[M-1,0,i] - diffs[M-1,0,i]


	# back side edge along the right face (back means away from origin)
	for i in range(1,P-1):
		val = sprec*(shrunk[M-1,N-1,i-1] + shrunk[M-1,N-1,i+1])
		val += sprec*(shrunk[M-2,N-1,i] + shrunk[M-1,N-2,i])

		val += eprec*(shrunk[M-1,N-2,i-1] + shrunk[M-1,N-2,i+1])
		val += eprec*(shrunk[M-2,N-1,i-1] + shrunk[M-2,N-1,i+1])
		val += eprec*shrunk[M-2,N-2,i]

		val += dprec*(shrunk[M-2,N-2,i-1] + shrunk[M-2,N-2,i+1])
		val += lprec*obs[M-1,N-1,i]

		shrunk[M-1,N-1,i] = val/prec
		diffs[M-1,N-1,i] = shrunk[M-1,N-1,i] - diffs[M-1,N-1,i]


	#------------------------------------------------

	prec = 5.0*sprec + 8.0*eprec + 4.0*dprec + lprec

	# bottom face
	for i in range(1,M-1):
		for j in range(1,N-1):
			val = sprec*(shrunk[i-1,j,0] + shrunk[i+1,j,0])
			val += sprec*(shrunk[i,j-1,0] + shrunk[i,j+1,0])
			val += sprec*shrunk[i,j,1]

			val += eprec*(shrunk[i-1,j-1,0] + shrunk[i+1,j-1,0])
			val += eprec*(shrunk[i-1,j+1,0] + shrunk[i+1,j+1,0])
			val += eprec*(shrunk[i-1,j,1] + shrunk[i+1,j,1])
			val += eprec*(shrunk[i,j-1,1] + shrunk[i,j+1,1])

			val += dprec*(shrunk[i-1,j-1,1] + shrunk[i+1,j-1,1])
			val += dprec*(shrunk[i-1,j+1,1] + shrunk[i+1,j+1,1])

			val += lprec*obs[i,j,0]

			shrunk[i,j,0] = val/prec
			diffs[i,j,0] = shrunk[i,j,0] - diffs[i,j,0]


	# top face
	for i in range(1,M-1):
		for j in range(1,N-1):
			val = sprec*(shrunk[i-1,j,P-1] + shrunk[i+1,j,P-1])
			val += sprec*(shrunk[i,j-1,P-1] + shrunk[i,j+1,P-1])
			val += sprec*shrunk[i,j,P-2]

			val += eprec*(shrunk[i-1,j-1,P-1] + shrunk[i+1,j-1,P-1])
			val += eprec*(shrunk[i-1,j+1,P-1] + shrunk[i+1,j+1,P-1])
			val += eprec*(shrunk[i-1,j,P-2] + shrunk[i+1,j,P-2])
			val += eprec*(shrunk[i,j-1,P-2] + shrunk[i,j+1,P-2])

			val += dprec*(shrunk[i-1,j-1,P-2] + shrunk[i+1,j-1,P-2])
			val += dprec*(shrunk[i-1,j+1,P-2] + shrunk[i+1,j+1,P-2])

			val += lprec*obs[i,j,P-1]

			shrunk[i,j,P-1] = val/prec
			diffs[i,j,P-1] = shrunk[i,j,P-1] - diffs[i,j,P-1]


	# left face
	for i in range(1,M-1):
		for k in range(1,P-1):
			val = sprec*(shrunk[i-1,0,k] + shrunk[i+1,0,k])
			val += sprec*(shrunk[i,0,k-1] + shrunk[i,0,k+1])
			val += sprec*shrunk[i,1,k]

			val += eprec*(shrunk[i-1,0,k-1] + shrunk[i+1,0,k-1])
			val += eprec*(shrunk[i-1,0,k+1] + shrunk[i+1,0,k+1])
			val += eprec*(shrunk[i-1,1,k] + shrunk[i+1,1,k])
			val += eprec*(shrunk[i,1,k-1] + shrunk[i,1,k+1])

			val += dprec*(shrunk[i-1,1,k-1] + shrunk[i+1,1,k-1])
			val += dprec*(shrunk[i-1,1,k+1] + shrunk[i+1,1,k+1])

			val += lprec*obs[i,0,k]

			shrunk[i,0,k] = val/prec
			diffs[i,0,k] = shrunk[i,0,k] - diffs[i,0,k]


	# right face
	for i in range(1,M-1):
		for k in range(1,P-1):
			val = sprec*(shrunk[i-1,N-1,k] + shrunk[i+1,N-1,k])
			val += sprec*(shrunk[i,N-1,k-1] + shrunk[i,N-1,k+1])
			val += sprec*shrunk[i,N-2,k]

			val += eprec*(shrunk[i-1,N-1,k-1] + shrunk[i+1,N-1,k-1])
			val += eprec*(shrunk[i-1,N-1,k+1] + shrunk[i+1,N-1,k+1])
			val += eprec*(shrunk[i-1,N-2,k] + shrunk[i+1,N-2,k])
			val += eprec*(shrunk[i,N-2,k-1] + shrunk[i,N-2,k+1])

			val += dprec*(shrunk[i-1,N-2,k-1] + shrunk[i+1,N-2,k-1])
			val += dprec*(shrunk[i-1,N-2,k+1] + shrunk[i+1,N-2,k+1])

			val += lprec*obs[i,N-1,k]

			shrunk[i,N-1,k] = val/prec
			diffs[i,N-1,k] =shrunk[i,N-1,k] - diffs[i,N-1,k]


	# back face
	for j in range(1,N-1):
		for k in range(1,P-1):
			val = sprec*(shrunk[0,j-1,k] + shrunk[0,j+1,k])
			val += sprec*(shrunk[0,j,k-1] + shrunk[0,j,k+1])
			val += sprec*shrunk[1,j,k]

			val += eprec*(shrunk[0,j-1,k-1] + shrunk[0,j+1,k-1])
			val += eprec*(shrunk[0,j-1,k+1] + shrunk[0,j+1,k+1])
			val += eprec*(shrunk[1,j-1,k] + shrunk[1,j+1,k])
			val += eprec*(shrunk[1,j,k-1] + shrunk[1,j,k+1])

			val += dprec*(shrunk[1,j-1,k-1] + shrunk[1,j+1,k-1])
			val += dprec*(shrunk[1,j-1,k+1] + shrunk[1,j+1,k+1])

			val += lprec*obs[0,j,k]

			shrunk[0,j,k] = val/prec
			diffs[0,j,k] = shrunk[0,j,k] - diffs[0,j,k]


	# front face
	for j in range(1,N-1):
		for k in range(1,P-1):
			val = sprec*(shrunk[M-1,j-1,k] + shrunk[M-1,j+1,k])
			val += sprec*(shrunk[M-1,j,k-1] + shrunk[M-1,j,k+1])
			val += sprec*shrunk[M-2,j,k]

			val += eprec*(shrunk[M-1,j-1,k-1] + shrunk[M-1,j+1,k-1])
			val += eprec*(shrunk[M-1,j-1,k+1] + shrunk[M-1,j+1,k+1])
			val += eprec*(shrunk[M-2,j-1,k] + shrunk[M-2,j+1,k])
			val += eprec*(shrunk[M-2,j,k-1] + shrunk[M-2,j,k+1])

			val += dprec*(shrunk[M-2,j-1,k-1] + shrunk[M-2,j+1,k-1])
			val += dprec*(shrunk[M-2,j-1,k+1] + shrunk[M-2,j+1,k+1])

			val += lprec*obs[M-1,j,k]

			shrunk[M-1,j,k] = val/prec

			diffs[M-1,j,k] = shrunk[M-1,j,k] - diffs[M-1,j,k]


	#------------------------------------------------

	# middle
	prec = 6.0*sprec + 12.0*eprec + 8.0*dprec
	prec = prec + lprec


	for i in range(1,M-1):
		for j in range(1,N-1):
			for k in range(1,P-1):
				val = sprec*(shrunk[i,j,k-1] + shrunk[i,j,k+1])
				val += sprec*(shrunk[i,j-1,k] + shrunk[i,j+1,k])
				val += sprec*(shrunk[i-1,j,k] + shrunk[i+1,j,k])

				val += eprec*(shrunk[i,j-1,k-1] + shrunk[i,j+1,k-1])
				val += eprec*(shrunk[i-1,j,k-1] + shrunk[i+1,j,k-1])

				val += eprec*(shrunk[i-1,j-1,k] + shrunk[i-1,j+1,k])
				val += eprec*(shrunk[i+1,j-1,k] + shrunk[i+1,j+1,k])

				val += eprec*(shrunk[i,j-1,k+1] + shrunk[i,j+1,k+1])
				val += eprec*(shrunk[i-1,j,k+1] + shrunk[i+1,j,k+1])

				val += dprec*(shrunk[i-1,j-1,k-1]+shrunk[i-1,j+1,k-1])
				val += dprec*(shrunk[i+1,j-1,k-1]+shrunk[i+1,j+1,k-1])
				val += dprec*(shrunk[i-1,j-1,k+1]+shrunk[i-1,j+1,k+1])
				val += dprec*(shrunk[i+1,j-1,k+1]+shrunk[i+1,j+1,k+1])

				val += lprec*obs[i,j,k]

				shrunk[i,j,k] = val/prec

				diffs[i,j,k] = shrunk[i,j,k] - diffs[i,j,k]

	print('max diff =', np.abs(diffs).max())

	return np.asarray(shrunk)
