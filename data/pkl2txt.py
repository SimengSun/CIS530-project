import glob, os
import pickle
for filename in glob.glob("*.pkl"):
    print(filename)
    with open(filename,'rb') as f:
	    data = pickle.load(f)
	    print(filename)
	    if 'en.pkl' == filename or 'es-en.pkl' == filename:
	    	print("len(data['train'])", list(data ))
	    	print("len(data['val'])", list(data))
	    	D = {}
	    	D[1] = []
	    	D[2] = []
	    	D['score'] = []
	    	for idx, s in enumerate(data['score']):
	    		if not -1 == s:
	    			D[1].append(data[1][idx])
	    			D[2].append(data[2][idx])
	    			D['score'].append(data['score'][idx])
    		N= len(D['score']) 
    		n1 = 6*int(N/10)
    		n2 = 8*int(N/10)
    		n3 = N
    		print('n1',n1,'n2',n2,'n3',n3)
	    	filewrite = filename[:-4]+'-train.txt'
	    	with open(filewrite,'w') as f_write:
	    		for i in range(n1):
	    			f_write.write( D[1][i]+'\t'+ D[2][i] +'\t'+str(D['score'][i]) + '\n')
	    	filewrite = filename[:-4]+'-val.txt'
	    	with open(filewrite,'w') as f_write:
	    		# D = data['val']
	    		for i in range(n1,n2):
	    			f_write.write( D[1][i]+'\t'+ D[2][i] +'\t'+str(D['score'][i]) + '\n')
	    	filewrite = filename[:-4]+'-test.txt'
	    	with open(filewrite,'w') as f_write:
	    		# D = data['val']
	    		for i in range(n2,n3):
	    			f_write.write( D[1][i]+'\t'+ D[2][i] +'\t'+str(D['score'][i]) + '\n')

	    print("---------------")