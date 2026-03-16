#
# This is evaluation func for Travel Agent Project. 
#
from pydantic import BaseModel
from typing import Any, List, Literal, Optional
from openai import OpenAI
import json
import sys

sys.path.append("/Users/hhung/Desktop/udacity_agentic_ai/")

from utils import (
    print_in_box,
    MODEL,
    OpenAIModels,
)
from utils_travel_agent import (
    Activity,
    ActivityRecommendation,
    ChatAgent,
    TravelPlan,
    VacationInfo,
    Weather,
    call_activity_by_id_api_mocked,
    do_chat_completion,
)
from test_data_travel_agent import (
    get_test_vacation_info,
    get_test_travel_plan,
)

with open('/Users/hhung/Desktop/udacity_agentic_ai/api_key.json', 'r') as file:
    api_keys = json.load(file)
    openai_key = api_keys["openai"]
    LLM_CLIENT = OpenAI(api_key=openai_key)


# Use an LLM to determine whether an event should be avoided due to weather conditions.
ACTIVITY_AND_WEATHER_ARE_COMPATIBLE_SYSTEM_PROMPT = """

You are a professional travel plan assistant. Given a travel plan, your job is to judge if the activities in the 
itinerary are suitable for weather conditions.

## Task

- If there is no enough info, e.g. no weather data is available, return IS_COMPATIBLE. 
- If there is weather data, identify if activities are suitable for the weather conditions:
    * If looks good, return IS_COMPATIBLE.
    * If not, return IS_INCOMPATIBLE, and also provided backup activities.

## Output format

    REASONING:
    - Look for each acvitity in the day-by-day plan
    - Check the weather data for the date.
    - Determine if the weather condition is suitable. "Unsuitable" means the activities may cause safety issue or hurt travelers. 
      For example it is dangerous to kayak in a rainny day, or long-time outdoor hiking at high-temperature.
    - As long as there is a unsitable activity, you should identify IS_INCOMPATIBLE and provide suggested activities. Otherwise IS_COMPATIBLE.

    FINAL ANSWER:
    [IS_COMPATIBLE, IS_INCOMPATIBLE]

## Examples
- If the weather is storm, any hiking is not suitable. You will return IS_INCOMPATIBLE. Meanwhile, suggest indoor activities as backup option.
- If the weather is clear, outdoor activities are suitable. You will return IS_COMPATIBLE.
""".strip()

TRAVELER_FEEDBACK = "I want to have at least two activities per day."


# Helper functions for running the evaluation functions
class AgentError(Exception):
    pass


class EvaluationResults(BaseModel):
    success: bool
    failures: List[str]
    eval_functions: List[str]


def get_eval_results(vacation_info, final_output, eval_functions) -> EvaluationResults:
    """
    Evaluates the final output of the itinerary agent against a set of evaluation functions.
    Args:
        vacation_info (VacationInfo): The vacation information used to generate the itinerary.
        final_output (TravelPlan): The final output from the itinerary agent.
        eval_functions (List[callable]): A list of evaluation functions to apply.
    Returns:
        EvaluationResults: An object containing the success status, any failures, and the names of the evaluation functions used.
    """
    # from project_lib import print_in_box
    if not isinstance(vacation_info, VacationInfo):
        raise ValueError("vacation_info must be an instance of VacationInfo")
    if not isinstance(final_output, TravelPlan):
        raise ValueError("final_output must be an instance of TravelPlan")
    if not isinstance(eval_functions, list) or not all(
        callable(fn) for fn in eval_functions
    ):
        raise ValueError("eval_functions must be a list of callable functions")

    eval_results = []
    for eval_fn in eval_functions:
        try:
            eval_fn(vacation_info, final_output)
        except AgentError as e:
            error_msg = str(e)
            print_in_box(error_msg, title="Evaluation Error")
            print("\n\n")

            eval_results.append(error_msg)
    return EvaluationResults(
        success=len(eval_results) == 0,
        failures=eval_results,
        eval_functions=[fn.__name__ for fn in eval_functions],
    )


# Basic evaluation functions
def eval_start_end_dates_match(vacation_info: VacationInfo, final_output: TravelPlan) -> None:
    """Verifies that the arrival and departure dates in vacation_info match the start and end dates in final_output.

    Args:
        vacation_info (dict): Contains the vacation details including arrival and departure dates
        final_output (dict): Contains the itinerary details including start and end dates

    Raises:
        AgentError: If either the arrival date doesn't match the start date or the departure date doesn't match the end date
    """
    if (
        vacation_info.date_of_arrival != final_output.start_date
        or vacation_info.date_of_departure != final_output.end_date
    ):
        raise AgentError(
            f"Dates do not match: {vacation_info.date_of_arrival} != {final_output.start_date} or {vacation_info.date_of_departure} != {final_output.end_date}"
        )

    if final_output.start_date > final_output.end_date:
        raise AgentError(
            f"Start date is after end date: {final_output.start_date} > {final_output.end_date}"
        )


# Evaluation functions related to the budget and total cost
def eval_total_cost_is_accurate(vacation_info: VacationInfo, final_output: TravelPlan) -> None:
    """Verifies that the total cost stated in final_output matches the sum of all activity prices.

    Args:
        vacation_info (dict): Contains the vacation details
        final_output (dict): Contains the itinerary details including activities with prices and total cost

    Raises:
        AgentError: If the calculated total cost doesn't match the stated total cost
    """
    actual_total_cost = 0

    for itinerary_day in final_output.itinerary_days:
        for activity_recommendation in itinerary_day.activity_recommendations:
            actual_total_cost += activity_recommendation.activity.price

    stated_total_cost = int(final_output.total_cost)

    if actual_total_cost != stated_total_cost:
        raise AgentError(
            f"Stated total cost does not match calculated total cost: {actual_total_cost} != {stated_total_cost}"
        )


def eval_total_cost_is_within_budget(vacation_info: VacationInfo, final_output: TravelPlan) -> None:
    """Verifies that the total cost stated in final_output is within the budget specified in vacation_info.

    Args:
        vacation_info (dict): Contains the vacation details including budget
        final_output (dict): Contains the itinerary details including total cost

    Raises:
        AgentError: If the total cost exceeds the budget
    """
    stated_total_cost = int(final_output.total_cost)
    if stated_total_cost > vacation_info.budget:
        raise AgentError(
            f"Total cost exceeds budget: {stated_total_cost} > {vacation_info.budget}"
        )


# The final output contains copies of the activities, so we need to verify that the copies are accurate
# and not hallucinated.
def eval_itinerary_events_match_actual_events(
    vacation_info: VacationInfo, final_output: TravelPlan
) -> None:
    """Verifies that the events listed in the itinerary match the actual events

    Args:
        vacation_info (dict): Contains the vacation details including traveler information and their interests
        final_output (dict): Contains the itinerary details including daily activities

    Raises:
        AgentError: If any traveler has no matching activities or if one traveler has more than twice
                   the number of matching activities compared to another traveler
    """
    event_ids_not_matching = []
    event_ids_missing = []

    for itinerary_day in final_output.itinerary_days:
        for activity_recommendation in itinerary_day.activity_recommendations:
            event_id = activity_recommendation.activity.activity_id
            # Assuming get_event_by_id is a function that retrieves the event by its ID

            reference_event = call_activity_by_id_api_mocked(event_id)

            if reference_event is None:
                event_ids_missing.append(event_id)

            elif Activity(**reference_event) != activity_recommendation.activity:
                print(
                    "---\n"
                    f"Event ID {event_id} does not match the reference event:\n"
                    f"Reference Event: {reference_event}\n"
                    f"Activity Event: {activity_recommendation.activity.model_dump()}"
                )
                event_ids_not_matching.append(event_id)
            else:
                # The event matches, so we can continue
                pass

    if event_ids_missing or event_ids_not_matching:
        raise AgentError(
            f"Event IDs missing: {event_ids_missing}\nEvent IDs not matching: {event_ids_not_matching}"
        )


# Check that the itinerary includes at least one activity matching each traveler's interests.
def eval_itinerary_satisfies_interests(
    vacation_info: VacationInfo, final_output: TravelPlan
) -> None:
    """Verifies that the itinerary includes activities matching each traveler's interests.

    This function checks that each traveler has at least one activity in the itinerary that matches their interests.

        Args:
        vacation_info (dict): Contains the vacation details including traveler information and their interests
        final_output (dict): Contains the itinerary details including daily activities

    Raises:
        AgentError: If any traveler has no matching activities or if one traveler has more than twice
                   the number of matching activities compared to another traveler
    """
    traveler_to_interests = {}
    traveler_to_interest_hit_counts = {}

    for traveler in vacation_info.travelers:
        traveler_to_interests[traveler.name] = traveler.interests
        traveler_to_interest_hit_counts[traveler.name] = 0

    for traveler_name, interests in traveler_to_interests.items():
        for itinerary_day in final_output.itinerary_days:
            for activity_recommendation in itinerary_day.activity_recommendations:
                # Check if the activity matches any of the traveler's interests
                matching_interests = set(traveler_to_interests[traveler_name]) & set(
                    activity_recommendation.activity.related_interests
                )

                if matching_interests:
                    traveler_to_interest_hit_counts[traveler_name] += 1
                    print(
                        f"✅ Traveler {traveler_name} has a match with interest {matching_interests} at {activity_recommendation.activity.name}"
                    )

    travelers_with_no_interest_hits = [
        traveler
        for traveler, interest_hit_count in traveler_to_interest_hit_counts.items()
        if interest_hit_count == 0
    ]

    # If any of the travelers have 0 matches, raise an error
    if travelers_with_no_interest_hits:
        raise AgentError(
            f"Travelers {travelers_with_no_interest_hits} has no matches with the itinerary."
        )


def eval_activities_and_weather_are_compatible(
    vacation_info: VacationInfo, 
    final_output: TravelPlan
) -> None:
    """Verifies that no outdoor-only activities are scheduled during inclement weather conditions.

    Args:
        vacation_info (dict): Contains the vacation details
        final_output (dict): Contains the itinerary details including daily activities and weather conditions

    Raises:
        AgentError: If any outdoor activities are scheduled during weather conditions that could ruin them
    """
    activities_that_are_incompatible = []

    for itinerary_day in final_output.itinerary_days:
        weather_condition = itinerary_day.weather.condition

        for activity_recommendation in itinerary_day.activity_recommendations:
            resp = do_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": ACTIVITY_AND_WEATHER_ARE_COMPATIBLE_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": f"Activity: {activity_recommendation.activity.name}\nDescription: {activity_recommendation.activity.description}\nWeather Condition: {weather_condition}",
                    },
                ],
                client=LLM_CLIENT,
                # This is a high-frequency use case, so we use a fast and cheap model.
                model=MODEL,
            )

            if "IS_COMPATIBLE" in (resp or ""):
                is_compatible = True
            elif "IS_INCOMPATIBLE" in (resp or ""):
                is_compatible = False
            else:
                raise RuntimeError(
                    f"Unexpected response from the model: {resp}. Expected 'IS_COMPATIBLE' or 'IS_INCOMPATIBLE'."
                )

            if is_compatible:
                print(
                    f"✅ Activity {activity_recommendation.activity.name} (on {itinerary_day.date}) and weather '{weather_condition}' are compatible."
                )

            else:
                activities_that_are_incompatible.append(
                    activity_recommendation.activity.name
                )
                print(
                    f"❌ Activity {activity_recommendation.activity.name} (on {itinerary_day.date}) and weather '{weather_condition}' are incompatible."
                )

    if activities_that_are_incompatible:
        raise AgentError(
            f"Activities that may be ruined by inclement weather: {activities_that_are_incompatible}"
        )


# Get the traveler's feedback and create a new evaluation function to check if the feedback was incorporated.
def eval_traveler_feedback_is_incorporated(
    vacation_info: VacationInfo, final_output: TravelPlan
) -> None:
    """Checks if the traveler's feedback was incorporated into the revised travel plan.

    Args:
        vacation_info (VacationInfo): The vacation information.
        final_output (TravelPlan): The revised travel plan.

    Raises:
        AgentError: If the traveler's feedback was not successfully incorporated.
    """

    agent = ChatAgent(
        system_prompt="""You are an expert in evaluating whether a travel plan incorporates traveler feedback.

    ## Output Format

    Respond using two sections (ANALYSIS AND FINAL OUTPUT) in the following format:

        ANALYSIS:
        * [step-by-step analysis]


        FINAL OUTPUT:
        [FULLY_INCORPORATED, PARTIALLY_INCORPORATED, NOT_INCORPORATED, or UNKNOWN]
        REASON: [reasoning for the final output]

    """,
        client=LLM_CLIENT,
        model=OpenAIModels.GPT_41,  # Use a powerful model for checking traveler feedback
    )

    resp = agent.chat(
        f"""Traveler Feedback: {TRAVELER_FEEDBACK}
    Revised Travel Plan: {final_output.model_dump_json()}
    """,
    )
    if "FINAL OUTPUT:" not in resp:
        raise RuntimeError(
            f"Unexpected response from the model: {resp}. Expected 'FINAL OUTPUT:'."
        )
    if "FULLY_INCORPORATED" not in resp:
        final_output = resp.split("FINAL OUTPUT:")[-1].strip()
        raise AgentError(
            f"Traveler feedback was not successfully incorporated into the revised travel plan. Response: {final_output}"
        )


if __name__ == "__main__":

    test_vacation_info = get_test_vacation_info()
    test_travel_plan = get_test_travel_plan()

    print(
        get_eval_results(
            vacation_info=test_vacation_info,
            final_output=test_travel_plan,
            eval_functions=[eval_start_end_dates_match],
        )
    )

    print(
        get_eval_results(
            vacation_info=test_vacation_info,
            final_output=test_travel_plan,
            eval_functions=[eval_total_cost_is_accurate, eval_total_cost_is_within_budget],
        )
    )

    print(
        get_eval_results(
            vacation_info=test_vacation_info,
            final_output=test_travel_plan,
            eval_functions=[eval_itinerary_events_match_actual_events],
        )
    )

    print(
        get_eval_results(
            vacation_info=test_vacation_info,
            final_output=test_travel_plan,
            eval_functions=[eval_itinerary_satisfies_interests],
        )
    )

    print(
        get_eval_results(
            vacation_info=test_vacation_info,
            final_output=test_travel_plan,
            eval_functions=[eval_activities_and_weather_are_compatible],
        )
    )

    all_eval_functions = [
        eval_start_end_dates_match,
        eval_total_cost_is_accurate,
        eval_itinerary_events_match_actual_events,
        eval_itinerary_satisfies_interests,
        eval_total_cost_is_within_budget,
        eval_activities_and_weather_are_compatible,
        eval_traveler_feedback_is_incorporated,
    ]

    print(
        get_eval_results(
            vacation_info=test_vacation_info,
            final_output=test_travel_plan,
            eval_functions=all_eval_functions,
        )
    )
    # NOTE for test_travel_plan, we should the following error message:
    # EvaluationResults(success=False, failures=['Traveler feedback was not successfully incorporated into the revised travel plan. 
    # Response: NOT_INCORPORATED\nREASON: The travel plan only schedules one activity per day and does not provide at least two activities 
    # per day as requested by the traveler’s feedback.'], eval_functions=['eval_start_end_dates_match', 'eval_total_cost_is_accurate', 
    # 'eval_itinerary_events_match_actual_events', 'eval_itinerary_satisfies_interests', 'eval_total_cost_is_within_budget', 
    # 'eval_activities_and_weather_are_compatible', 'eval_traveler_feedback_is_incorporated'])
