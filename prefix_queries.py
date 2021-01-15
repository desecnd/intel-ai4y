from subprocess import Popen, PIPE
from sys import platform

def runProcess(dictionaryName):

	if platform == 'win32':
		binaryFullName = 'words_prediction\\trie_words_predictor.exe '
	elif platform == 'linux' or platform == 'linux2':
		binaryFullName = 'words_prediction/trie_words_predictor '
	else:
		raise SystemError('Operating System is not recognized, make sure you are running under Linux or Windows OS')
		
	print(binaryFullName)
	processHandle = Popen(binaryFullName + dictionaryName, stdin=PIPE,  stdout=PIPE, shell=True) 
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
