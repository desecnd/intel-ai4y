// ---- Authors: Pawel Wozniak, Ewelina Tyma
// -- Basic client for Trie Dictionary word prediction

#include <string>
#include <iostream>

#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#define PORT 9123

int main() {
	
	uint32_t MAXBUFFSIZE = 256;
	int sockfd, bytesRecv;
	char buffer[MAXBUFFSIZE] = {0};
	struct sockaddr_in serv_addr = {0};

	if((sockfd = socket(PF_INET, SOCK_STREAM, 0)) == -1) {
		std::cerr << "socket() function error\n";
		exit(EXIT_FAILURE);
	}

	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(PORT);
	serv_addr.sin_addr.s_addr = inet_addr("127.0.0.1");

	if( connect(sockfd, (struct sockaddr *) &serv_addr, sizeof(struct sockaddr)) == -1) {
		std::cerr << "connect() function error\n";
		exit(EXIT_FAILURE);
	}
	
	while (1) {
		std::cout << "Enter prefix for prediction: ";
		std::string message;
		std::cin >> message;

		send(sockfd, message.c_str(), message.size(), 0); 

		if(( bytesRecv = recv(sockfd, buffer, MAXBUFFSIZE-1, 0)) == -1) {
			std::cerr << "recv() function error\n";
			exit(EXIT_FAILURE);
		}

		std::string predictions (buffer, bytesRecv);
		std::cout << "[PREDICTIONS]: " << predictions << '\n';
	}
	
	return 0;
}
