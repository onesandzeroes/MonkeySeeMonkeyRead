import text.blob as blob
import text.tokenizers as tokenizers


class CharTokenizer(tokenizers.BaseTokenizer):

    def tokenize(self, text):
        all_chars = []
        words = blob.TextBlob(text).words
        for w in words:
            all_chars.extend(list('^' + w + '$'))
        return all_chars


eg_text = "Textblob is amazingly simple to use. What great fun!"
bigrams = blob.TextBlob(eg_text, tokenizer=CharTokenizer())