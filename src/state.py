from typing import Annotated, List, Optional, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class HotelAgentState(TypedDict):
    # add_messages ავტომატურად ამატებს ახალ მესიჯებს ისტორიის ბოლოში
    messages: Annotated[List[BaseMessage], add_messages]
    room_number: Optional[str]
    guest_name: Optional[str]
    pending_order: Optional[Dict[str, Any]]
    tool_error_count: int  # ინახავს tool-ის ერორების რაოდენობას