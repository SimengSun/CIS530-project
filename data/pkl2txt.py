import glob, os
import pickle
for filename in glob.glob("*.pkl"):
    print(filename)
    with open(filename,'rb') as f:
	    data = pickle.load(f)
	    print(filename)
	    if 'score' in list(data):
	    	print('succeed')
	    	# filewrite = filename[:-4] + '.txt'
	    	# with open(filewrite,'w') as f_write:
	    	# 	D = data
	    	# 	for i in range(len(D['score'])):
	    	# 		f_write.write( D[1][i]+'\t'+ D[2][i] +'\t'+str(D['score'][i]) + '\n')
	    else:  
	    	print("len(data['train'])", list(data['train'] ) )
	    	print("len(data['val'])", list(data['val'] ) )
	    	data['train'][1] +=  data['val'][1]
	    	data['train'][2] +=  data['val'][2]
	    	data['train']['score'] +=  data['val']['score']
	    	D = {}
	    	D[1] = []
	    	D[2] = []
	    	D['score'] = []
	    	for idx, s in enumerate(data['train']['score']):
	    		if not -1 == s:
	    			D[1].append(data['train'][1][idx])
	    			D[2].append(data['train'][2][idx])
	    			D['score'].append(data['train']['score'][idx])
    		N= len(D['score']) 
    		n1 = 6*int(N/10)
    		n2 = 8*int(N/10)
    		n3 = N
    		print('n1',n1,'n2',n2,'n3',n3)
	    	filewrite = filename[6:-4]+'-train.txt'
	    	with open(filewrite,'w') as f_write:
	    		for i in range(n1):
	    			f_write.write( D[1][i]+'\t'+ D[2][i] +'\t'+str(D['score'][i]) + '\n')
	    	filewrite = filename[6:-4]+'-val.txt'
	    	with open(filewrite,'w') as f_write:
	    		# D = data['val']
	    		for i in range(n1,n2):
	    			f_write.write( D[1][i]+'\t'+ D[2][i] +'\t'+str(D['score'][i]) + '\n')
	    	filewrite = filename[6:-4]+'-test.txt'
	    	with open(filewrite,'w') as f_write:
	    		# D = data['val']
	    		for i in range(n2,n3):
	    			f_write.write( D[1][i]+'\t'+ D[2][i] +'\t'+str(D['score'][i]) + '\n')

	    print("---------------")