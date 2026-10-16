import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from src.state import HotelAgentState
from src.tools import order_room_service, get_wifi_info

# 1. LLM და Tool-ების დაკავშირება
tools = [order_room_service, get_wifi_info]
tools_by_name = {t.name: t for t in tools}

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
).bind_tools(tools)

SYSTEM_PROMPT = """შენ ხარ ქართულენოვანი სასტუმროს ხმოვანი ასისტენტი.
შენი მიზანია დაეხმარო სტუმარს თავაზიანად და მოკლედ (რადგან პასუხი ხმით უნდა წაიკითხო).
თუ სტუმარი ითხოვს სერვისს ან მენიუს, გამოიყენე შესაბამისი ხელსაწყო (Tool).
თუ ხელსაწყომ დააბრუნა შეცდომა, თავაზიანად აუხსენი სტუმარს და შესთავაზე ალტერნატივა."""

# --- კვანძები (NODES) ---

def agent_node(state: HotelAgentState):
    """LLM იღებს გადაწყვეტილებას"""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def tool_node(state: HotelAgentState):
    """Tool-ების გაშვება და შეცდომების (Error) დაჭერა"""
    last_message = state["messages"][-1]
    new_messages = []
    error_count = state.get("tool_error_count", 0)
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        selected_tool = tools_by_name.get(tool_name)
        
        try:
            # ვცდილობთ Tool-ის გაშვებას
            observation = selected_tool.invoke(tool_args)
            new_messages.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
            )
        except Exception as e:
            # თუ ჩავარდა: არ ვაქრაშებთ პროგრამას!
            error_count += 1
            error_msg = f"Tool Failure: {str(e)}"
            new_messages.append(
                ToolMessage(content=error_msg, tool_call_id=tool_call["id"])
            )
            
    return {"messages": new_messages, "tool_error_count": error_count}

def human_escalation_node(state: HotelAgentState):
    """როცა შეცდომა მეორდება, ზარი გადადის ადამიანთან"""
    msg = SystemMessage(
        content="სისტემური შეფერხებაა. გთხოვთ, სტუმარს უთხრათ: 'ბოდიშს გიხდით, სისტემასთან კავშირი შეფერხდა. თქვენს ზარს გადავამისამართებ სასტუმროს ადმინისტრატორთან.'"
    )
    res = llm.invoke([msg])
    return {"messages": [res]}

# --- პირობითი ლოგიკა (CONDITIONAL EDGES) ---

def route_after_agent(state: HotelAgentState):
    last_message = state["messages"][-1]
    # თუ LLM-ს უნდა ხელსაწყოს გამოყენება
    if last_message.tool_calls:
        return "tools"
    return END

def route_after_tools(state: HotelAgentState):
    # თუ შეცდომები დაგროვდა (მაგ: 2 ან მეტი ჩავარდნა)
    if state.get("tool_error_count", 0) >= 2:
        return "human_escalation"
    # სხვა შემთხვევაში ვაბრუნებთ აგენტთან, რომ შედეგი შეაფასოს
    return "agent"

# --- გრაფის აწყობა ---
builder = StateGraph(HotelAgentState)

builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_node("human_escalation", human_escalation_node)

builder.set_entry_point("agent")

builder.add_conditional_edges("agent", route_after_agent, {
    "tools": "tools",
    END: END
})

builder.add_conditional_edges("tools", route_after_tools, {
    "human_escalation": "human_escalation",
    "agent": "agent"
})

builder.add_edge("human_escalation", END)

# გრაფის დაკომპილირება
hotel_graph = builder.compile()