import re
from typing import Iterator

from langchain_core.messages import HumanMessage
from src.graph import hotel_graph

# წინადადების დასასრული: სასვენი ნიშანი ან ახალი ხაზი
SENTENCE_END = re.compile(r"(?<=[.!?\n])")


def stream_sentences(user_text: str, room_number: str = "304") -> Iterator[str]:
    """LLM-ის ტოკენებს აგროვებს და TTS-ისთვის აბრუნებს სრულ წინადადებებს, როგორც კი მზად იქნება."""
    input_state = {
        "messages": [HumanMessage(content=user_text)],
        "room_number": room_number,
        "tool_error_count": 0,
    }

    buffer = ""
    for msg_chunk, metadata in hotel_graph.stream(input_state, stream_mode="messages"):
        # მხოლოდ სტუმრისთვის განკუთვნილი ტექსტი (Tool-ის შედეგები არ უნდა გაჟღერდეს)
        if metadata.get("langgraph_node") not in ("agent", "human_escalation"):
            continue
        # .text აბრუნებს მხოლოდ ტექსტურ ბლოკებს (thinking და tool_use ბლოკები გამოტოვებულია)
        if not msg_chunk.text:
            continue

        buffer += msg_chunk.text

        # ერთ chunk-ში შეიძლება რამდენიმე წინადადება დასრულდეს
        *complete, buffer = SENTENCE_END.split(buffer)
        for sentence in complete:
            if sentence.strip():
                yield sentence.strip()

    if buffer.strip():
        yield buffer.strip()


if __name__ == "__main__":
    for sentence in stream_sentences("დილა მშვიდობისა, ყავა მინდა ოთახში და როდის იქნება მზად?"):
        print(f"🗣️ [TTS]: {sentence}")
