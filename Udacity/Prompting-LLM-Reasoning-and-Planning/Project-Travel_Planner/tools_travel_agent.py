
from utils_travel_agent import (
    Activity,
    TravelPlan,
    VacationInfo,
    call_activity_by_id_api_mocked,
    call_activities_api_mocked,
)
from eval_travel_agent import (
    get_eval_results,
    eval_activities_and_weather_are_compatible,
    eval_itinerary_events_match_actual_events,
    eval_itinerary_satisfies_interests,
    eval_start_end_dates_match,
    eval_total_cost_is_within_budget,
    eval_total_cost_is_accurate,
    eval_traveler_feedback_is_incorporated,
)
from test_data_travel_agent import (
    get_test_vacation_info,
    get_test_travel_plan,
)


ALL_EVAL_FUNCTIONS = [
    eval_start_end_dates_match,
    eval_total_cost_is_accurate,
    eval_itinerary_events_match_actual_events,
    eval_itinerary_satisfies_interests,
    eval_total_cost_is_within_budget,
    eval_activities_and_weather_are_compatible,
    eval_traveler_feedback_is_incorporated,
]


# Helper function to generate tool descriptions from function docstrings
def get_tool_descriptions_string(fns):
    """Generates a tool description from a function's docstring.
    Args:
        fns (list): List of functions to generate descriptions for.
    Returns:
        str: A formatted string containing the function names and their descriptions."""
    resp = ""
    for fn in fns:
        function_name = fn.__name__
        function_doc = fn.__doc__ or "No description provided."
        resp += f"* `{function_name}`: {function_doc}\n"

    return resp


# Define the calculator tool that evaluates mathematical expressions.
def calculator_tool(input_expression: str) -> float:
    """Evaluates a mathematical expression and returns the result as a float.

    Args:
        input_expression (str): A string containing a valid mathematical expression to evaluate.

    Returns:
        float: The result of the evaluated expression.

    Example:
        >>> calculator_tool("1 + 1")
        2.0
    """
    import numexpr as ne
    return float(ne.evaluate(input_expression))


# Tool to fetch activities for a given date and city.
def get_activities_by_date_tool(date: str, city: str) -> list[dict]:
    """
    Retrieve activities given a date and a city names.

    Args:
        date (str): date, yyyy-mm-dd
        city (str): city name

    Returns:
        list[dict]: The list of activity dictionary.

    """
    resp = call_activities_api_mocked(date=date, city=city)
    return [Activity.model_validate(activity).model_dump() for activity in resp]


# Tool to run all evaluation functions on a travel plan.
def run_evals_tool(vacation_info: VacationInfo, travel_plan: TravelPlan) -> dict:
    """Runs all evaluation tools on the provided travel plan and vacation info.

    Args:
        travel_plan (TravelPlan): The travel plan to evaluate.

    Returns:
        EvaluationResults: The results of the evaluations.
    """
    if isinstance(travel_plan, dict):
        travel_plan = TravelPlan.model_validate(travel_plan)

    resp = get_eval_results(
        vacation_info=vacation_info,
        final_output=travel_plan,
        eval_functions=ALL_EVAL_FUNCTIONS,
    )
    return {
        # Show the success status and any failures
        "success": resp.success,
        "failures": resp.failures,
    }


# A tool to return the final travel plan
def final_answer_tool(final_output: TravelPlan) -> TravelPlan:
    """Returns the final travel plan

    Args:
        final_output (TravelPlan): The final travel plan to return.

    Returns:
        TravelPlan: The final travel plan.
    """
    return final_output


ALL_TOOLS = [
    calculator_tool,
    get_activities_by_date_tool,
    run_evals_tool,
    final_answer_tool,
]


if __name__ == "__main__":
    assert calculator_tool("1 + 1") == 2.0
    print(get_tool_descriptions_string([calculator_tool]))

    assert len(get_activities_by_date_tool("2025-06-10", "AgentsVille")) > 0
    print(get_tool_descriptions_string([get_activities_by_date_tool]))

    print(get_tool_descriptions_string([run_evals_tool]))

    # Let's double check that the tool works as expected.
    test_vacation_info = get_test_vacation_info()
    test_travel_plan = get_test_travel_plan()    
    run_evals_tool(vacation_info=test_vacation_info, travel_plan=test_travel_plan)

    print(get_tool_descriptions_string([final_answer_tool]))
