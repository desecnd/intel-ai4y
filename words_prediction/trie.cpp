#include <iostream>
#include <chrono>
#include <memory>
#include <vector>
#include <string>
#include <fstream>

struct Node {
	std::vector<std::shared_ptr<Node>> childs;	
	std::shared_ptr<Node> parent;
	char letter;
	bool eow; 

	Node (std::shared_ptr<Node> p, char x, bool endOfWord = false) : childs(), parent(p), letter(x), eow(endOfWord) {}
};


struct Trie{
	std::shared_ptr<Node> root;		
			
	Trie() : root(std::make_shared<Node>(nullptr, '*', false)) {}

	void loadFromFile(const std::string & name) { 
		std::ifstream dictionary(name);
	
		if (!dictionary.good()) {
			std::cerr << "Error, cannot open file " << name << '\n';
			return;
		}
		
		std::string word;
		while (dictionary >> word) {
			this->addWord(word);
		}
	}
	
	void addWord(const std::string & word) {
		std::shared_ptr<Node> ptr = root;

		for (char x : word) { 
			std::shared_ptr<Node> child;
			for ( auto ch : ptr->childs ) { 
				if ( ch->letter == x ) child = ch;  						
			}

			if ( !child ) {
				child = std::make_shared<Node>(ptr, x, false);
				ptr->childs.push_back(child);
			}
			ptr = child;
		}

		ptr->eow = true;
	}

	std::vector<std::string> prefixSearch(const std::string & prefix, int n) {
		std::shared_ptr<Node> ptr = root;	

		for (char x : prefix ) {
							
			std::shared_ptr<Node> child;
			for ( auto ch : ptr->childs ) { 
				if ( ch->letter == x ) child = ch;  						
			}
				
			if ( !child ) 
				return std::vector<std::string>();

			ptr = child; 
		}

		std::vector<std::shared_ptr<Node>> ends;
		this->addEndWords(ptr, ends, n);

		std::vector<std::string> predictions; 
		for ( auto wordEnd : ends ) {
			predictions.push_back( this->recurseString( wordEnd ) );
		}
		
		return predictions;
	}

	void addEndWords(std::shared_ptr<Node> curr, std::vector<std::shared_ptr<Node>> & ends, int & wordsLeft) {
		if ( wordsLeft <= 0 ) return;

		else if ( curr->eow ) {
			ends.push_back(curr);
			wordsLeft--;
		}

		for ( auto child : curr->childs ) {
			addEndWords(child, ends, wordsLeft);
		}
	}

	std::string recurseString(std::shared_ptr<Node> ptr) {
		if ( ptr == root ) return ""; 	
		else return recurseString(ptr->parent) + ptr->letter;
	} 

	void printTrie(std::shared_ptr<Node> curr, int level = 0) {
		std::cout <<  std::string(level, '-') << curr->letter << (curr->eow ? '$' : ' ')  << "\n";
		for (auto ptr : curr->childs) {
				
			printTrie(ptr, level + 1);
		}
	}
} trie;

int main(int argc, char ** argv) {
	std::ios::sync_with_stdio(false); 
	size_t nOfWords	= 5;
	std::string dictionaryName = "";	
	
	if (argc == 2) {
		dictionaryName = std::string(argv[1]);   
	}
	else if (argc == 3) {
		dictionaryName = std::string(argv[1]);
		nOfWords = std::atoi(argv[2]); 
	}
	else  {
		std::cerr << "Usage: program_name dictionary_path n_of_searched_words\n";
		return 1;
	}
	

	// auto start = std::chrono::steady_clock::now();
	trie.loadFromFile(dictionaryName);

	// auto end = std::chrono::steady_clock::now();
	// std::cout << "Loading time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " milliseconds\n";
	// trie.printTrie(trie.root);
	
	std::string prefix;
	while (true) {
		// auto start = std::chrono::steady_clock::now();

		std::cin >> prefix;	
		std::vector<std::string> predictions { trie.prefixSearch(prefix, nOfWords) };

		std::string predictedWords = "";
		for ( std::string pred : predictions ) {
			predictedWords += pred + " ";
		}

		std::cout << predictedWords << '\n';
		std::cout.flush();

		//auto end = std::chrono::steady_clock::now();
		// std::cout << "Query time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds\n";
	
	}
			
	return 0;
}

