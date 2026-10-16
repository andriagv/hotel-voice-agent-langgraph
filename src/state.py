from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class HotelAgentState(TypedDict):
    # add_messages ავტომატურად ამატებს ახალ მესიჯებს ისტორიის ბოლოში
    messages: Annotated[List[BaseMessage], add_messages]
    room_number: Optional[str]
    guest_name: Optional[str]

    # ქვემოთ მოცემული მთვლელები ყოველ ახალ შეკითხვაზე ნულდება (input_guard)
    tool_error_count: int            # tool-ის ჩავარდნების რაოდენობა
    agent_steps: int                 # LLM-ის გამოძახებები ამ შეკითხვაზე (ციკლის ლიმიტი)
    verify_attempts: int             # verifier-მა რამდენჯერ დააბრუნა პასუხი გადასაწერად
    verifier_feedback: Optional[str] # რატომ უარყო verifier-მა ბოლო პასუხი
