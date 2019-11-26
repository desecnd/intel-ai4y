// ---- Authors: Pawel Wozniak, Ewelina Tyma

#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <vector>
#include <chrono>

#include <sys/socket.h> // Socket, Listen, Bind, etc
#include <netinet/in.h> // htonl, htons, etc
#include <unistd.h>  // close etc	
#include <arpa/inet.h>	// inet_ntoa()

#define PORT 9123
#define BACKLOG 10

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

void runServer() {

	int32_t sockfd, clientfd, SockOptEnabled = 1;
	struct sockaddr_in serv_addr {0}, client_addr {0};
	uint32_t sin_size = sizeof(struct sockaddr_in);
	uint32_t BUFFERSIZE = 32;

	if((sockfd = socket(PF_INET, SOCK_STREAM, 0)) == -1) {
		std::cerr << "socket() function error\n";
		exit(EXIT_FAILURE);
	}

	if( setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &SockOptEnabled, sizeof(int)) == -1) {
		std::cerr << "setsockopt() function error\n";
		exit(EXIT_FAILURE);
	}

	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(PORT);
	serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);

	if( bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(struct sockaddr)) == -1) {
		std::cerr << "bind() function error\n";
		exit(EXIT_FAILURE);
	}

	if( listen(sockfd , BACKLOG) == -1) {
		std::cerr << "listen() function error\n";
		exit(EXIT_FAILURE);
	}


	std::cout << "|-- TCP/IP Trie Dictionary Server is running on port " << PORT << " --|\n";

	while (1) {
		if((clientfd = accept(sockfd, (struct sockaddr *) &client_addr, &sin_size)) == -1) {
				std::cerr << "accept() function error\n";
				exit(EXIT_FAILURE);
		}

		std::cout << "[CONNECTED]: " << inet_ntoa(client_addr.sin_addr) << '\n';

		while(1) {
			char buffer[BUFFERSIZE] {0};
			int32_t bytesRead = recv( clientfd, buffer, BUFFERSIZE, 0);
		
			if (bytesRead == 0) {
				std::cout << "Client left\n";	
				close(clientfd);
				break;
			} else if (bytesRead == -1) {
				std::cerr << "recv() function error\n";	
				exit(EXIT_FAILURE);
			} else {
				std::string prefix(buffer, bytesRead);
				std::cout << "Prefix received: " << prefix << '\n';

				auto start = std::chrono::steady_clock::now();
				std::vector<std::string> predictions { trie.prefixSearch(prefix, 5) };
				auto end = std::chrono::steady_clock::now();
				std::cout << "Query time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds\n";

				std::string returnMessage = "";
				for ( std::string pred : predictions ) 
					returnMessage += pred + " ";

				send(clientfd, returnMessage.c_str(), returnMessage.size(), 0);
			}
		}
	}
}

int main() {
	//std::ios::sync_with_stdio(false); 
	
	auto start = std::chrono::steady_clock::now();
	trie.loadFromFile("polish_dictionary.txt");
	auto end = std::chrono::steady_clock::now();
	
	std::cout << "Loading time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " milliseconds\n";

	// trie.printTrie(trie.root);

	runServer();
	return 0;
}

