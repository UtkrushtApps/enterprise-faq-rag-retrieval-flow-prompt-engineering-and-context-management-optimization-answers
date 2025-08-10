import tiktoken

class Tokenizer:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.encoding = tiktoken.encoding_for_model(model)

    def encode(self, text: str):
        return self.encoding.encode(text)

    def count(self, text: str):
        return len(self.encode(text))
