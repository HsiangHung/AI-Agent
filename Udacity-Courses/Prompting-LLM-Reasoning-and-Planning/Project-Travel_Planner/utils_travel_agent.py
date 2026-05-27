import json
from enum import Enum
from pydantic import BaseModel, Field  # For structured data validation
from typing import List, Literal, Optional
import datetime
import sys

sys.path.append("/Users/hhung/Desktop/udacity_agentic_ai/")
from utils import (
    get_completion,
    print_in_box,
)

from data_travel_agent import (
    ACTIVITY_CALENDAR,
    INCLIMATE_WEATHER_CONDITIONS,
    WEATHER_FORECAST,
)


class Interest(str, Enum):
    ART = "art"
    COOKING = "cooking"
    COMEDY = "comedy"
    DANCING = "dancing"
    FITNESS = "fitness"
    GARDENING = "gardening"
    HIKING = "hiking"
    MOVIES = "movies"
    MUSIC = "music"
    PHOTOGRAPHY = "photography"
    READING = "reading"
    SPORTS = "sports"
    TECHNOLOGY = "technology"
    THEATRE = "theatre"
    TENNIS = "tennis"
    WRITING = "writing"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class Traveler(BaseModel):
    """A traveler with a name, age, and list of interests.
    
    Attributes:
        name (str): The name of the traveler.
        age (int): The age of the traveler.
        interests (List[Interest]): A list of interests of the traveler.
    """
    name: str
    age: int
    interests: list[Interest]


class VacationInfo(BaseModel):
    """Vacation information including travelers, destination, dates, and budget.
    Attributes:
        travelers (List[Traveler]): A list of travelers.
        destination (str): The vacation destination.
        date_of_arrival (datetime.date): The date of arrival.
        date_of_departure (datetime.date): The date of departure.
        budget (int): The budget for the vacation in fictional currency units.
    """
    travelers: list[Traveler]
    destination: str = Field(..., min_length=2, max_length=50)
    date_of_arrival: datetime.date
    date_of_departure: datetime.date
    budget: int


def call_weather_api_mocked(date: str, city: str) -> dict[str, str | int]:
    """
    Returns the weather forecast for a given date and city.

    Args:
        date: The date to get weather for. Must be in the format YYYY-MM-DD.
        city: The city to get weather for.

    Returns:
        A dictionary containing the weather forecast for the given date and city.
    """
    # If the city is not AgentsVille, return an empty dictionary
    if city != "AgentsVille":
        return {}

    # Verify the date format
    try:
        datetime.datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        print(f"Invalid date format: {date}")
        return {}

    # If the date is not between 2025-06-10 and 2025-06-15, return an empty dictionary
    if date < "2025-06-10" or date > "2025-06-15":
        print(f"Date {date} is outside the valid range (2025-06-10 - 2025-06-15)")
        return {}

    return next(
        (forecast for forecast in WEATHER_FORECAST if forecast["date"] == date), {}
    )


def call_activities_api_mocked(
    date: str | None = None, city: str | None = None, activity_ids: list[str] | None = None
) -> list[dict[str, str | int]]:
    """Calls the mocked activities API to get a list of activities for a given date and city.

    This function simulates an API call to retrieve activities based on the provided date and city.

    Args:
        date: The date to get activities for. Must be in the format YYYY-MM-DD.
        city: The city to get activities for. Only "AgentsVille" is supported.
        activity_ids: A list of activity IDs to filter the results. If None, all activities for the date and city will be returned.

    Returns:
        A list of activities for the given date and city. Currently only returns activities
        for AgentsVille between 2025-06-10 and 2025-06-15.
    """
    import datetime

    # If the city is not AgentsVille, return an empty list
    if city and city != "AgentsVille":
        return []

    # Verify the date format
    if date:
        try:
            datetime.datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            print(f"Invalid date format: {date}")
            return []

    # If the date is not between 2025-06-10 and 2025-06-15, return an empty list
    if date and (date < "2025-06-10" or date > "2025-06-15"):
        print(f"Date {date} is outside the valid range (2025-06-10 - 2025-06-15)")
        return []

    activities = ACTIVITY_CALENDAR

    if date:
        activities = [event for event in activities if event["start_time"].startswith(date)]

    if activity_ids:
        activities = [event for event in activities if event["activity_id"] in activity_ids]

    if not activities:
        print(f"No activities found for {date} in {city}.")
    return activities


def call_activity_by_id_api_mocked(activity_id: str):
    """Calls the mocked activity API to get an activity by its ID.

    Args:
        activity_id: The ID of the event to retrieve.

    Returns:
        A dictionary containing the event details, or an empty dictionary if not found.
    """
    for event in ACTIVITY_CALENDAR:
        if event["activity_id"] == activity_id:
            return event
    print(f"Event with ID {activity_id} not found.")
    return None


class Weather(BaseModel):
    temperature: float
    temperature_unit: str
    condition: str


class Activity(BaseModel):
    activity_id: str
    name: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    location: str
    description: str
    price: int
    related_interests: List[Interest]


class ActivityRecommendation(BaseModel):
    activity: Activity
    reasons_for_recommendation: List[str]


class ItineraryDay(BaseModel):
    date: datetime.date
    weather: Weather
    activity_recommendations: List[ActivityRecommendation]


class TravelPlan(BaseModel):
    city: str
    start_date: datetime.date
    end_date: datetime.date
    total_cost: int
    itinerary_days: List[ItineraryDay]


def do_chat_completion(messages: list[dict[str, str]], model=None, client=None, **kwargs):
    """A simple wrapper around OpenAI's chat completion API.

    Args:
        messages: A list of messages to send to the chat completion API.

    Returns:
        str: The response from the chat completion API.

    Raises:
        openai.OpenAIError: If the chat completion API returns an error.

    Examples:
        >>> messages = [
        ...     {"role": "user", "content": "Hello, how are you?"},
        ...     {"role": "assistant", "content": "I'm good, thanks!"},
        ... ]
        >>> from unittest.mock import patch
        >>> with patch('openai.OpenAI') as mock_openai:
        ...     # Setup mock response
        ...     mock_client = mock_openai.return_value
        ...     mock_chat = mock_client.chat
        ...     mock_completions = mock_chat.completions
        ...     mock_create = mock_completions.create
        ...     mock_response = mock_create.return_value
        ...     mock_response.choices = [type('obj', (object,), {'message': type('msg', (object,), {'content': "I'm good, thanks!"})()})]
        ...     response = do_chat_completion(messages)
        >>> response
        "I'm good, thanks!"
    """
    if client is None:
        raise ValueError("A valid OpenAI client must be provided.")
    
    if model is None:
        raise ValueError("A valid model must be provided.")

    if "response_format" not in kwargs:
        response = client.chat.completions.create(  # type: ignore
            model=model,
            messages=messages,  # type: ignore
            **kwargs,  # type: ignore
        )
    else:
        response = client.beta.chat.completions.parse(  # type: ignore
            model=model,
            messages=messages,  # type: ignore
            **kwargs,  # type: ignore
        )

    if hasattr(response, "error"):
        raise RuntimeError(
            f"OpenAI API returned an error: {str(response.error)}"
        )

    return response.choices[0].message.content


class ChatAgent:
    """A chat agent that interacts with OpenAI's API to facilitate conversations.

    This class manages chat history, formats system prompts, and handles
    communication with OpenAI's chat completion API. It provides methods to
    add messages, get responses, and maintain conversation context.

    Attributes:
        system_prompt_template (str): Template for the system prompt using {variable_name} placeholders.
    """
    system_prompt = "You are a helpful assistant."
    messages = []

    def __init__(self, name=None, system_prompt=None, client=None, model=None):
        self.name = name or self.__class__.__name__
        if system_prompt:
            self.system_prompt = system_prompt
        self.client = client
        self.model = model
        self.reset()

    def add_message(self, role, content):
        """Add a message to the chat history.

        Args:
            role (str): The role of the message ("system", "user", or "assistant").
            content (str): The content of the message.

        Raises:
            ValueError: If the role is not one of "system", "user", or "assistant".
        """
        if role not in ["system", "user", "assistant"]:
            raise ValueError(f"Invalid role: {role}")
        self.messages.append({"role": role, "content": content})
        if role == "system":
            print_in_box(
                content,
                f"{self.name} - System Prompt",
            )
        elif role == "user":
            print_in_box(
                content,
                f"{self.name} - User Prompt",
            )
        elif role == "assistant":
            print_in_box(
                content,
                f"{self.name} - Assistant Response",
            )

    def reset(self):
        """Reset the chat history and re-initialize with the system prompt.

        This method clears all existing messages and adds the system prompt
        formatted with the template_kwargs.
        """
        from textwrap import dedent

        system_prompt = dedent(self.system_prompt).strip()

        # Clear previous messages and add the system prompt
        self.messages = []
        self.add_message(
            "system",
            system_prompt,
        )

    def get_response(self, add_to_messages=True, model=None, client=None, **kwargs):
        """Get a response from the OpenAI API.

        Args:
            add_to_messages (bool, optional): Whether to add the response to the chat history
            using the add_message method and the assistant role. Defaults to True.

        Returns:
            str: The response from the OpenAI API.


        """
        response = do_chat_completion(
            messages=self.messages,
            model=model or self.model,
            client=client or self.client,
            **kwargs
        )
        if add_to_messages:
            self.add_message("assistant", response)
        return response

    def chat(self, user_message, add_to_messages=True, model=None, **kwargs):
        """Send a message to the chat and get a response.

        Args:
            user_message (str): The message to send to the chat.

        Returns:
            str: The response from the OpenAI API.
        """
        self.add_message("user", user_message)
        return self.get_response(add_to_messages=add_to_messages, model=model, **kwargs)
