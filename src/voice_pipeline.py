import re
import uuid
from typing import Any, Iterator, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from src.graph import hotel_graph

# წინადადების დასასრული: სასვენი ნიშანი ან ახალი ხაზი
SENTENCE_END = re.compile(r"(?<=[.!?\n])")

YES_WORDS = {"კი", "დიახ", "ხო", "ჰო", "კარგი", "დადასტურებული", "yes", "ok", "ki"}


def split_sentences(text: str) -> Iterator[str]:
    """ტექსტს ჭრის წინადადებებად, რომ TTS-მა თითოეული მზადებისთანავე გააჟღეროს"""
    for sentence in SENTENCE_END.split(text):
        if sentence.strip():
            yield sentence.strip()


def is_yes(answer: str) -> bool:
    words = re.findall(r"\w+", answer.lower())
    return any(word in YES_WORDS for word in words)


class GuestSession:
    """ერთი სტუმრის საუბარი: thread_id-ით გრაფი წინა შეკითხვებს იმახსოვრებს"""

    def __init__(self, room_number: str):
        self.room_number = room_number
        self.config = {"configurable": {"thread_id": f"room-{room_number}-{uuid.uuid4().hex[:8]}"}}
        self.pending_confirmation: Optional[dict] = None

    def respond(self, guest_text: str) -> Iterator[str]:
        """სტუმრის ნათქვამზე აბრუნებს გასაჟღერებელ წინადადებებს"""
        if self.pending_confirmation:
            # გრაფი შეკვეთის დადასტურებას ელოდება: ვაგრძელებთ იქიდან, სადაც გაჩერდა
            graph_input: Any = Command(resume=is_yes(guest_text))
        else:
            graph_input = {"messages": [HumanMessage(content=guest_text)], "room_number": self.room_number}

        result = hotel_graph.invoke(graph_input, self.config)

        # interrupt: გრაფი გაჩერდა და სტუმრის დადასტურებას ელოდება
        interrupts = result.get("__interrupt__")
        if interrupts:
            self.pending_confirmation = interrupts[0].value
            yield from split_sentences(self.pending_confirmation["question"])
            return

        self.pending_confirmation = None
        # verifier-ის გავლის შემდეგ ბოლო მესიჯი სტუმრისთვის განკუთვნილი პასუხია
        last_message = result["messages"][-1]
        if isinstance(last_message, AIMessage):
            yield from split_sentences(last_message.text)


if __name__ == "__main__":
    session = GuestSession(room_number="304")
    print("🏨 სასტუმროს ასისტენტი (ოთახი 304). გასასვლელად: Enter ცარიელ ხაზზე.\n")
    while True:
        guest_text = input("🎙️ სტუმარი: ").strip()
        if not guest_text:
            break
        for sentence in session.respond(guest_text):
            print(f"🗣️ [TTS]: {sentence}")
        print()
