from subprocess import Popen, PIPE

def runServer(dictionaryName):
	serv = Popen('words_prediction/trie_words_predictor ' + dictionaryName, stdin=PIPE,  stdout=PIPE, shell=True) 
	return serv

def queryServer(serv, prefix):
	if serv.poll():
		return "trie_words_predictor not running\n"
	
	serv.stdin.write(bytes(prefix + "\n", 'UTF-8'))
	serv.stdin.flush()


	output = serv.stdout.readline().decode('UTF-8')
	predictions = output.split()

	return predictions
	
def closeServer(serv):
	serv.terminate()
