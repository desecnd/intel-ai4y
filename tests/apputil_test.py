from utils import apputil

class TestAppendWord:
    def test_append_word_to_empty_message(self):
        current_message = ""
        word_to_append = "word"
        new_message = apputil.append_word(current_message, word_to_append)
        
        assert(new_message == word_to_append + " ")
        
    def test_append_word_to_word(self):
        current_message = "word1 "
        word_to_append = "word2"
        new_message = apputil.append_word(current_message, word_to_append)
        
        expected_message = current_message + word_to_append + " "
        assert(new_message == expected_message)

    def test_autocomplete_last_word(self):
        current_message = "my test mes"
        word_to_append = "message"
        new_message = apputil.append_word(current_message, word_to_append)

        expected_message = "my test message "
        assert(new_message == expected_message)

    def test_dont_autocomplete_last_word(self):
        current_message = "my test message"
        word_to_append = "message"
        new_message = apputil.append_word(current_message, word_to_append)

        expected_message = "my test message message "
        assert(new_message == expected_message)
