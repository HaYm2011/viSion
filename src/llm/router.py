import logging
from src.db.database import get_last_known_location

logger = logging.getLogger(__name__)

class QueryRouter:
    def __init__(self):
        # Initialize local LLM (e.g. llama.cpp or an API client)
        logger.info("Initialized QueryRouter")

    def route_query(self, query: str) -> str:
        """
        Determines if the query is asking about an object's location.
        If yes, fetches from DB.
        If no, falls back to general LLM response.
        """
        query_lower = query.lower()

        # Super naive router for the skeleton
        if "where" in query_lower:
            # Extract object name naively
            words = query_lower.split()
            # e.g., "where are my keys" -> "keys"
            object_name = words[-1].strip("?")

            return self._handle_location_query(object_name)
        else:
            return "I am an ambient memory assistant. I can tell you where your things are."

    def _handle_location_query(self, object_name: str) -> str:
        location_data = get_last_known_location(object_name)

        if location_data:
            zone = location_data["zone_id"]
            timestamp = location_data["timestamp"]
            # A real LLM would generate a natural sentence here
            return f"Your {object_name} was last seen at the {zone} at {timestamp}."
        else:
            return f"I haven't seen your {object_name} recently."

def get_answer(query: str) -> str:
    router = QueryRouter()
    return router.route_query(query)
