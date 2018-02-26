#!/usr/bin/python

import numpy as np
import cv2
import sys

def cross_prod(m, n):
	j, k = m.shape
	k, l = n.shape
	assert(m.shape[1] == n.shape[0])
	p = np.zeros((j,l), dtype=np.float32)
	for x in range(0, j):
		for y in range(0,l):
			print m[x]
			print n[:,y]
			m_v = m[x]
			n_v = np.transpose(n[:,y])
			for z in range(0,k):
				p[x,y] += int(m_v[z]) * int(n_v[z])
	return p
	
j = int(sys.argv[1])
k = int(sys.argv[2])
l = int(sys.argv[3])

mat_text = []
for lines in sys.stdin.xreadlines():
	mat_text.append(lines)

m = np.zeros((j,k), dtype=np.float64)
n = np.zeros((k,l), dtype=np.float64)

p_read = np.zeros((j,l), dtype=np.float64)


for x in range(0, j):
	line = mat_text.pop(0)
	m[x] = [int(float(nums.rstrip())) for nums in line.split(" ")[:-1]]

mat_text.pop(0)

for x in range(0, k):
	line = mat_text.pop(0)
	n[x] = [int(float(nums.rstrip())) for nums in line.split(" ")[:-1]]

mat_text.pop(0)

for x in range(0, j):
	line = mat_text.pop(0)
	p_read[x] = [float(nums.rstrip()) for nums in line.split(" ")[:-1]]


p = np.mat(m) * np.mat(n)

assert((j, k) == m.shape)
assert((k, l) == n.shape)
assert((j, l) == p.shape)
assert(p.shape == p_read.shape)

success =  np.all(np.isclose(p, p_read))
if success:
	print success
else:
	print n, "\n"
	print m, "\n"
	print p_read
	print "\n"
	print p

