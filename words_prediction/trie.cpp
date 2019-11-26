#include <bits/stdc++.h>

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
			std::cout << "Error, cannot open file " << name << '\n';
			return;
		}
		
		std::cout << "Adding words started\n";

		std::string word;
		while (dictionary >> word) {
			this->addWord(word);
		}
		std::cout << "Adding words finished\n";
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

int main() {
	std::ios::sync_with_stdio(false); 
	
	auto start = std::chrono::steady_clock::now();
	trie.loadFromFile("polish_dictionary.txt");
	auto end = std::chrono::steady_clock::now();
	
	std::cout << "Loading time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " milliseconds\n";

	// trie.printTrie(trie.root);
	
	std::string prefix;
	while (1) {
		std::cout << "Enter word prefix: ";
		std::cin >> prefix;	
		
		auto start = std::chrono::steady_clock::now();
		std::vector<std::string> predictions { trie.prefixSearch(prefix, 5) };
		auto end = std::chrono::steady_clock::now();
		std::cout << "Query time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds\n";
	
		int i = 1;
		for ( std::string pred : predictions ) {
			std::cout << i++ << ". " << pred << '\n';
		}
	
	}
			
	return 0;
}

