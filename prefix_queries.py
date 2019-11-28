from subprocess import Popen, PIPE

def runProcess(dictionaryName):
	processHandle = Popen('words_prediction/trie_words_predictor ' + dictionaryName, stdin=PIPE,  stdout=PIPE, shell=True) 
	return processHandle

def queryProcess(processHandle, prefix):
	if processHandle.poll():
		return "trie_words_predictor not running\n"
	
	processHandle.stdin.write(bytes(prefix + "\n", 'UTF-8'))
	processHandle.stdin.flush()


	output = processHandle.stdout.readline().decode('UTF-8')
	predictions = output.split()

	return predictions
	
def closeProcess(processHandle):
	processHandle.terminate()
