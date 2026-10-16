from typing import Annotated
from pydantic import Field
from langchain_core.tools import tool, InjectedToolArg

# იმიტირებული მონაცემთა ბაზა (PMS)
ROOM_SERVICE_MENU = {
    "ყავა": 8.0,
    "ომლეტი": 15.0,
    "წყალი": 3.0,
    "კრუასანი": 7.0
}

MAX_QUANTITY = 10


@tool
def order_room_service(
    item: str,
    room_number: Annotated[str, InjectedToolArg],
    quantity: Annotated[int, Field(ge=1, le=MAX_QUANTITY)] = 1,
) -> str:
    """სასტუმროს ნომერში საჭმლის ან სასმლის შეკვეთის ხელსაწყო.

    item: პროდუქტის სახელი მენიუდან. quantity: რაოდენობა (1-დან 10-მდე).
    """
    # room_number LLM-ს არ ეჩვენება (InjectedToolArg): მას State აწვდის,
    # ასე რომ სტუმარი სხვა ოთახის სახელზე შეკვეთას ვერ გააფორმებს
    item_clean = item.strip().lower()

    # სპეციალურად ჩავაშენოთ შეცდომის ლოგიკა ტესტირებისთვის
    if "ხინკალი" in item_clean:
        raise ValueError(f"შეცდომა: 'ხინკალი' არ არის ნომრის სერვისის მენიუში. ხელმისაწვდომია: {list(ROOM_SERVICE_MENU.keys())}")

    if item_clean in ROOM_SERVICE_MENU:
        price = ROOM_SERVICE_MENU[item_clean] * quantity
        return f"შეკვეთა მიღებულია: {quantity}x {item_clean}, ოთახი {room_number}. ჯამური ფასი: {price} ლარი."
    else:
        return f"პროდუქტი '{item}' ვერ მოიძებნა მენიუში. ხელმისაწვდომია: {list(ROOM_SERVICE_MENU.keys())}"

@tool
def get_wifi_info() -> str:
    """სასტუმროს Wi-Fi პაროლის გაგება."""
    return "ქსელი: Hotel_Guest_5G, პაროლი: WelcomeToHotel2026"
