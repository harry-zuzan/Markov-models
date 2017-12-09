#from collections import namedtuple
#
##Segment = namedtuple('Segment',['start','stop','value'])
#
#class Segment:
#	def __init__(self,start,stop,value):
#		self.start = start
#		self.stop = stop
#		self.value = value
#
#	def size(self):
#		return self.stop - self.start
#
## A doubly linked list of segments
#class DLSeg:
#	def __init__(self,seg):
#		self.seg = seg
#		self.seg_prev = None
#		self.seg_next = None
#
#
#class DLSegList:
#	def __init__(self,seg_list,split_value):
#		self.dl_segs = [DLSeg(seg) for seg in seg_list]
#
#		self.split_value = split_value
#
#		# remove any segments not of interest from the ends
#		if len(self):
#			if self.dl_segs[0].seg.value == self.split_value:
#				self.dl_segs.pop(0)
#
#		if len(self):
#			if self.dl_segs[-1].seg.value == self.split_value:
#				self.dl_segs.pop()
#
#		# create the double linked list
#		for k in range(1,len(self)):
#			self.dl_segs[k-1].seg_next = self.dl_segs[k]
#			self.dl_segs[k].seg_prev = self.dl_segs[k-1]
#
#
#
#		max_len = 0
#
#		for dl_seg in self.dl_segs:
#			if not dl_seg.seg.value == split_value: continue
#			if not (dl_seg.seg.stop - dl_seg.seg.start) > max_len: continue
#			
#			max_len = dl_seg.seg.stop - dl_seg.seg.start
#			self.split_seg = dl_seg
#
#		return
#
#
## need to split a doubly linked list of segments
#	def __len__(self): return len(self.dl_segs)
#
#	def __getitem__(self,i): return self.dl_segs[i].seg
#
#	def size(self):
#		if not len(self): return 0
#		return self.stop() - self.start()
#
#	def start(self):
#		if not self.dl_segs: return None
#		return self.dl_segs[0].seg.start
#
#	def stop(self):
#		if not self.dl_segs: return None
#		return self.dl_segs[-1].seg.stop
#
#	def size(self):
#		if not len(self): return 0
#		return self.stop() - self.start()
#
#
#	def trace_sum(self):
#		the_sum = 0
#		for dl_seg in self.dl_segs:
#			if dl_seg.seg.value == self.split_value: continue
#			the_sum += dl_seg.seg.size()
#
#		return the_sum
#
#	def trace_prop(self):
#		return float(self.trace_sum())/float(self.size())
#			
#
#	def split(self):
#		if not len(self): return (None,None)
#
#		segs_left = []
#		segs_right = []
#
#		idx = 0
#		while idx < len(self):
#			if not self.dl_segs[idx] == self.split_seg:
#				segs_left.append(self.dl_segs[idx].seg)
#				idx += 1
#				continue
#			idx += 1
#			break
#
#		while idx < len(self):
#			segs_right.append(self.dl_segs[idx].seg)
#			idx += 1
#
#
#		dl_list_left,dl_list_right = None,None
#		if segs_left: dl_list_left = DLSegList(segs_left,self.split_value)
#		if segs_right: dl_list_right = DLSegList(segs_right,self.split_value)
#
#
#		return dl_list_left,dl_list_right
#
#	def segments(self):
#		segs = [dl_seg.seg for dl_seg in self.dl_segs]
#		return segs
#
#
#def segment_values(vec):
#	N = len(vec)
#
#	segment_list = []
#
#	idx = 0
#	seg_val = vec[0]
#	seg_start = idx
#	seg_stop = idx
#
#	while idx < N:
#		if vec[idx] == seg_val:
#			idx += 1
#			continue
#
#		seg_stop = idx
#
#		seg = Segment(seg_start,seg_stop,seg_val)
#		segment_list.append(seg)
#
#		seg_val = vec[idx]
#		seg_start = idx
#
#	seg_stop = idx
#
#	seg = Segment(seg_start,seg_stop,seg_val)
#	segment_list.append(seg)
#
#	return segment_list
#
