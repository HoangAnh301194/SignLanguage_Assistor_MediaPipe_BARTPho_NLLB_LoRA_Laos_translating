from src.nlp.sentence_buffer import SentenceBuffer
from src.nlp.sentence_builder import SentenceBuilder


def test_sentence_buffer():
    buffer = SentenceBuffer()

    buffer.add_word("toi")
    buffer.add_word("muon")
    buffer.add_word("nuoc")

    assert buffer.raw_sentence() == "toi muon nuoc"


def test_sentence_builder_rule():
    builder = SentenceBuilder(use_bartpho=False)

    assert builder.build("toi muon nuoc") == "Tôi muốn uống nước."
    assert builder.build("toi can giup do") == "Tôi cần giúp đỡ."
    assert builder.build("cam_on") == "Cảm ơn bạn."


if __name__ == "__main__":
    test_sentence_buffer()
    test_sentence_builder_rule()
    print("All NLP tests passed.")
