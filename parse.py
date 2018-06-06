
f=open("bigrams.txt","r")
g=open("bigramsEdited.txt","w")
import pdb
l=[]
for line in f:
	line=line.strip()
	if len(line)>3:
		#pdb.set_trace()
		d=eval(line.split("\t")[1])
		m=max(d.values())
		d={key:val for key,val in d.items() if val!=m}
		if (m/max(d.values())>2):
			g.write(line+"\n")
l.sort()
for index,elem in enumerate(l):
	print("{}: {}".format(index,elem))
f.close()
g.close()
