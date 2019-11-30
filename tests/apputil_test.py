from utils import apputil

def test_append_word_to_empty_message():
    current_message = ""
    word_to_append = "word"
    new_message = apputil.append_word(current_message, word_to_append)
    
    assert(new_message == word_to_append + " ")
    
def test_append_word_to_word():
    current_message = "word1 "
    word_to_append = "word2"
    new_message = apputil.append_word(current_message, word_to_append)
    
    expected_message = current_message + word_to_append + " "
    assert(new_message == expected_message)
