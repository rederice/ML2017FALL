import sys
filename = sys.argv[1]
f = open(filename,'r').readline()
ff = f.strip()
array = (ff).split(' ')
seen = []
same = []
for one in array:
	if one in seen:
		pass
	else:
		same.append(array.count(one))
		seen.append(one)
file = open("Q1.txt",'w')
for i in range(len(seen)):
	tmpline = str(seen[i])+' '+str(i)+' '+str(same[i])
	if(i!=len(seen)-1):
		tmpline += '\n'
	file.write(tmpline)