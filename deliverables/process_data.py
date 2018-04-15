import codecs

pairs_file = codecs.open("../data/all-pairs.txt", "r")
gs_file = codecs.open("../data/all-gs.txt", "r")

outfile = codecs.open("../data/all-data.txt", "w")


pair_lines = pairs_file.readlines()
gs_lines = gs_file.readlines()

for i in range(len(pair_lines)):
	sentences = pair_lines[i].split("\t")
	score = gs_lines[i].strip()
	if (len(sentences) < 2 or score == ""):
		continue
	else:
		sent1 = sentences[0].strip()
		sent2 = sentences[1].strip()
		outfile.write(sent1 + "\t" + sent2 + "\t" + score + "\n")
