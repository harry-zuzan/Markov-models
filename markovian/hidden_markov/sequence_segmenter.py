from collections import namedtuple

#Segment = namedtuple('Segment',['start','stop','value'])

class Segment:
	def __init__(self,start,stop,value):
		self.start = start
		self.stop = stop
		self.value = value

	def size(self):
		return self.stop - self.start

# A doubly linked list of segments
class DLSeg:
	def __init__(self,seg):
		self.seg = seg
		self.dl_prev = None
		self.dl_next = None
		self.reducible = True
		


def min_seg_size(seg_list,min_sz):
	dl_segs = []

	for seg in seg_list:
		dl_segs.append(DLSeg(seg))

	for k in range(1,len(dl_segs)):
		dl_segs[k-1].dl_next = dl_segs[k]
		dl_segs[k].dl_prev = dl_segs[k-1]

	working_list = []

	for dl_seg in dl_segs:
		if dl_seg.seg.size() < min_sz: continue
		dl_seg.reducible = False
		working_list.append(dl_seg)

	working_list.sort(key=lambda x: -x.seg.size())

	while working_list:
		push_back = False
		dl_seg = working_list.pop(0)

		if dl_seg.dl_prev:
			if dl_seg.dl_prev.reducible:
				push_back = True
				dl_seg.seg.start -= 1
				dl_seg.dl_prev.seg.stop -= 1
				if not dl_seg.dl_prev.seg.size():
					seg_list.remove(dl_seg.dl_prev.seg)
					dl_seg.dl_prev = dl_seg.dl_prev.dl_prev
					if dl_seg.dl_prev:
						dl_seg.dl_prev.dl_next = dl_seg

		if dl_seg.dl_next:
			if dl_seg.dl_next.reducible:
				push_back = True
				dl_seg.seg.stop += 1
				dl_seg.dl_next.seg.start += 1
				if not dl_seg.dl_next.seg.size():
					seg_list.remove(dl_seg.dl_next.seg)
					dl_seg.dl_next = dl_seg.dl_next.dl_next
					if dl_seg.dl_next:
						dl_seg.dl_next.dl_prev = dl_seg

		if push_back: working_list.append(dl_seg)


def get_segments(vec):
	N = len(vec)

	segment_list = []

	idx = 0
	seg_val = vec[0]
	seg_start = idx
	seg_stop = idx

	while idx < N:
		if vec[idx] == seg_val:
			idx += 1
			continue

		seg_stop = idx

		seg = Segment(seg_start,seg_stop,seg_val)
		segment_list.append(seg)

		seg_val = vec[idx]
		seg_start = idx

	seg_stop = idx

	seg = Segment(seg_start,seg_stop,seg_val)
	segment_list.append(seg)

	return segment_list

