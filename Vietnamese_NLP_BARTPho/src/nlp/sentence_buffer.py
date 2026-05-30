import time


class SentenceBuffer:
    """
    Lưu các từ/cụm từ được nhận diện theo thứ tự.
    Có chức năng chống lặp từ trong realtime.
    """

    def __init__(self, repeat_cooldown: float = 1.5):
        self.words = []
        self.last_word = None
        self.last_time = 0.0
        self.repeat_cooldown = repeat_cooldown

    def add_word(self, word: str):
        word = word.strip()

        if word in ["", "unknown", "uncertain"]:
            return

        now = time.time()

        if word == self.last_word and now - self.last_time < self.repeat_cooldown:
            return

        self.words.append(word)
        self.last_word = word
        self.last_time = now

    def remove_last(self):
        if self.words:
            self.words.pop()

    def clear(self):
        self.words.clear()
        self.last_word = None
        self.last_time = 0.0

    def raw_sentence(self) -> str:
        return " ".join(self.words)

    def get_words(self):
        return self.words.copy()
