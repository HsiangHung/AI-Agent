import datetime
from utils_travel_agent import (
    Activity,
    ActivityRecommendation,
    ItineraryDay,
    VacationInfo,
    Traveler,
    TravelPlan,
    Weather,
)


def get_test_vacation_info():
    vacation_info_dict = {
        "travelers": [
            {
                "name": "Yuri",
                "age": 30,
                "interests": ["tennis", "cooking", "comedy", "technology"],
            },
            {
                "name": "Hiro",
                "age": 25,
                "interests": ["reading", "music", "theatre", "art"],
            },
        ],
        "destination": "AgentsVille",
        "date_of_arrival": "2025-06-10",  # Mock data exists for 2025-06-10
        "date_of_departure": "2025-06-12",  # ...until 2025-06-15.
        "budget": 130,  # Budget is in fictional currency units.
    }
    # Validate the VacationInfo data structure
    vacation_info = VacationInfo.model_validate(vacation_info_dict)

    # Check that VacationInfo contains the expected data
    assert "travelers" in vacation_info.model_dump().keys(), "VacationInfo should contain 'travelers' key"
    assert "destination" in vacation_info.model_dump().keys(), "VacationInfo should contain 'destination' key"
    assert "date_of_arrival" in vacation_info.model_dump().keys(), "VacationInfo should contain 'date_of_arrival' key"
    assert "date_of_departure" in vacation_info.model_dump().keys(), "VacationInfo should contain 'date_of_departure' key"
    assert "budget" in vacation_info.model_dump().keys(), "VacationInfo should contain 'budget' key"
    assert isinstance(vacation_info.travelers, list), "Travelers should be a list"
    assert all([isinstance(traveler, Traveler) for traveler in vacation_info.travelers]), "All travelers should be instances of Traveler class"
    assert isinstance(vacation_info.date_of_arrival, datetime.date), "date_of_arrival should be a date"
    assert isinstance(vacation_info.date_of_departure, datetime.date), "date_of_departure should be a date"
    assert isinstance(vacation_info.budget, int), "budget should be an integer"

    # If all assertions pass, print a success message
    print("✅ VacationInfo data structure is valid!")

    return vacation_info


def get_test_travel_plan():
    """
    This test travel plan is for initial itinerary plan. Only ONE activity per day.
    So this travel plan will fail in "eval_traveler_feedback_is_incorporated".
    """
    travel_plan = TravelPlan(
        city="AgentsVille",
        start_date=datetime.date(2025, 6, 10),
        end_date=datetime.date(2025, 6, 12),
        total_cost=55,
        itinerary_days=[
            ItineraryDay(
                date=datetime.date(2025, 6, 10), 
                weather=Weather(temperature=31.0, temperature_unit="celsius", condition="clear"), 
                activity_recommendations=[
                    ActivityRecommendation(
                        activity=Activity(
                            activity_id="event-2025-06-10-1", 
                            name="Serve & Savor: Tennis and Taste Luncheon",
                            start_time=datetime.datetime(2025, 6, 10, 12, 0),
                            end_time=datetime.datetime(2025, 6, 10, 13, 30),
                            location="The Grand Racquet Terrace, AgentsVille",
                            description="Join us for 'Serve & Savor,' the ultimate crossover event for cooking and tennis enthusiasts in AgentsVille! Kick off your lunch hour with a friendly round of doubles on our outdoor courts, then unwind with a hands-on cooking workshop led by a local chef, where you'll prepare and enjoy delicious energy-boosting recipes. Whether you come for the sport or the flavors, this energizing luncheon celebrates both passions in a lively outdoor setting. Ideal for anyone who loves to play, cook, or simply savor fresh food and fun!", 
                            price=20, 
                            related_interests=["cooking", "tennis"]
                        ), 
                        reasons_for_recommendation=[
                            "Matches Yuri's interests in tennis and cooking perfectly.",
                            "Suitable for sunny clear weather to enjoy outdoor courts.",
                            "Engaging and active experience, promoting social interaction.",
                        ]
                    )
                ]
            ), 
            ItineraryDay(
                date=datetime.date(2025, 6, 11),
                weather=Weather(temperature=34.0, temperature_unit="celsius", condition="partly cloudy"),
                activity_recommendations=[
                    ActivityRecommendation(
                        activity=Activity(
                            activity_id="event-2025-06-11-1",
                            name="Tech Lunch & Learn: AI Frontiers",
                            start_time=datetime.datetime(2025, 6, 11, 12, 0),
                            end_time=datetime.datetime(2025, 6, 11, 13, 30),
                            location="The Digital Atrium, AgentsVille",
                            description="Join fellow tech enthusiasts for a dynamic lunchtime event exploring the future of artificial intelligence! Held indoors at The Digital Atrium, this Tech Lunch & Learn features engaging lightning talks, interactive demos, and networking opportunities all centered around technology and innovation. Enjoy light lunch fare as you connect with others passionate about technology, AI, and the digital world. Whether you're a seasoned developer or just curious about tech, this event is for you! Related interests: technology, music (sound tech demos), photography (AI imaging), writing (AI creativity).",
                            price=20,
                            related_interests=["technology", "music", "photography", "writing"]
                        ), 
                        reasons_for_recommendation=[
                            "Indoor event suitable for partly cloudy weather.",
                            "Matches Yuri's interest in technology and Hiro's interests in music, writing, and art.",
                            "Interactive and educational, offers networking for tech enthusiasts.",
                        ]
                    )
                ]
            ),
            ItineraryDay(
                date=datetime.date(2025, 6, 12),
                weather=Weather(temperature=28.0, temperature_unit='celsius', condition='thunderstorm'),
                activity_recommendations=[
                    ActivityRecommendation(
                        activity=Activity(
                            activity_id='event-2025-06-12-3',
                            name='Tech & Film Fusion Night',
                            start_time=datetime.datetime(2025, 6, 12, 19, 0),
                            end_time=datetime.datetime(2025, 6, 12, 21, 30),
                            location="Virtual Reality Theater, Silicon Plaza, AgentsVille",
                            description='Dive into an immersive evening where the magic of movies meets the latest in technology! Join fellow movie buffs and tech enthusiasts for a special screening of cutting-edge sci-fi short films, followed by an interactive panel with local filmmakers and VR technologists. Experience the future of entertainment and discuss how technology is transforming the world of cinema. This exciting, indoor event at the Virtual Reality Theater is perfect for anyone interested in technology and movies.',
                            price=15,
                            related_interests=["technology", "movies"]
                        ),
                        reasons_for_recommendation=[
                            'Indoor event ideal for thunderstorm weather.',
                            "Aligns with Yuri's interest in technology and Hiro's interest in movies.",
                            'Engaging night event combining tech and film for a unique experience.'
                        ]
                    )
                ]
            )
        ]
    )

    return travel_plan
