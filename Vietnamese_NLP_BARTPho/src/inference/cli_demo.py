from src.nlp.sentence_buffer import SentenceBuffer
from src.nlp.sentence_builder import SentenceBuilder


def run_cli():
    buffer = SentenceBuffer()
    builder = SentenceBuilder()

    print("Demo Language Core - VSL Vietnamese NLP")
    print("Nhập từ/cụm từ: toi, muon, nuoc")
    print("Lệnh: /build, /clear, /delete, /exit")

    while True:
        user_input = input("Nhập ký hiệu: ").strip()

        if user_input == "/exit":
            break

        if user_input == "/clear":
            buffer.clear()
            print("Đã xóa buffer.")
            continue

        if user_input == "/delete":
            buffer.remove_last()
            print("Buffer hiện tại:", buffer.raw_sentence())
            continue

        if user_input == "/build":
            raw = buffer.raw_sentence()
            sentence = builder.build(raw)

            print("Chuỗi thô:", raw)
            print("Câu hoàn chỉnh:", sentence)
            continue

        buffer.add_word(user_input)
        print("Buffer hiện tại:", buffer.raw_sentence())
