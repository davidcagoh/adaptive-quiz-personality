"""
MBTI schema loader.
Returns schema compatible with AdaptiveEngine.
"""


def load_mbti_schema() -> dict:
    """
    Load MBTI schema with dimensions [EI, SN, TF, JP] and question pool.

    Returns
    -------
    dict
        Schema compatible with AdaptiveEngine:
        {
            "dimensions": ["EI", "SN", "TF", "JP"],
            "questions": [{"id": str, "text": str, "weights": [float, float, float, float]}, ...]
        }
    """
    return {
        "dimensions": ["EI", "SN", "TF", "JP"],
        "questions": [
            {"id": "q1", "text": "You feel energized after spending time with a group of people.", "weights": [1.0, 0.0, 0.0, 0.0]},
            {"id": "q2", "text": "You prefer to recharge by spending time alone.", "weights": [-1.0, 0.0, 0.0, 0.0]},
            {"id": "q3", "text": "You enjoy meeting new people at social events.", "weights": [1.0, 0.0, 0.0, 0.0]},
            {"id": "q4", "text": "You prefer deep conversations with a few close friends over small talk with many.", "weights": [-1.0, 0.0, 0.0, 0.0]},
            {"id": "q5", "text": "You think out loud and process ideas by talking.", "weights": [1.0, 0.0, 0.0, 0.0]},
            {"id": "q6", "text": "You prefer to think things through internally before speaking.", "weights": [-1.0, 0.0, 0.0, 0.0]},
            {"id": "q7", "text": "You focus on concrete facts and what can be observed.", "weights": [0.0, 1.0, 0.0, 0.0]},
            {"id": "q8", "text": "You are drawn to abstract concepts and future possibilities.", "weights": [0.0, -1.0, 0.0, 0.0]},
            {"id": "q9", "text": "You trust your five senses and practical experience.", "weights": [0.0, 1.0, 0.0, 0.0]},
            {"id": "q10", "text": "You notice patterns and connections others miss.", "weights": [0.0, -1.0, 0.0, 0.0]},
            {"id": "q11", "text": "You prefer information that is specific and detailed.", "weights": [0.0, 1.0, 0.0, 0.0]},
            {"id": "q12", "text": "You enjoy exploring theoretical ideas and concepts.", "weights": [0.0, -1.0, 0.0, 0.0]},
            {"id": "q13", "text": "You make decisions based on objective logic and analysis.", "weights": [0.0, 0.0, 1.0, 0.0]},
            {"id": "q14", "text": "You consider people's feelings and values when making decisions.", "weights": [0.0, 0.0, -1.0, 0.0]},
            {"id": "q15", "text": "You prioritize truth and fairness over harmony.", "weights": [0.0, 0.0, 1.0, 0.0]},
            {"id": "q16", "text": "You value empathy and maintaining relationships.", "weights": [0.0, 0.0, -1.0, 0.0]},
            {"id": "q17", "text": "You analyze problems objectively before acting.", "weights": [0.0, 0.0, 1.0, 0.0]},
            {"id": "q18", "text": "You trust your gut feelings and personal values.", "weights": [0.0, 0.0, -1.0, 0.0]},
            {"id": "q19", "text": "You prefer structure, plans, and organization.", "weights": [0.0, 0.0, 0.0, 1.0]},
            {"id": "q20", "text": "You like flexibility and keeping your options open.", "weights": [0.0, 0.0, 0.0, -1.0]},
            {"id": "q21", "text": "You work best with clear deadlines and schedules.", "weights": [0.0, 0.0, 0.0, 1.0]},
            {"id": "q22", "text": "You prefer to go with the flow and adapt as you go.", "weights": [0.0, 0.0, 0.0, -1.0]},
            {"id": "q23", "text": "You like to finish tasks before starting new ones.", "weights": [0.0, 0.0, 0.0, 1.0]},
            {"id": "q24", "text": "You enjoy spontaneity and last-minute changes.", "weights": [0.0, 0.0, 0.0, -1.0]},
            {"id": "q25", "text": "You feel comfortable making decisions quickly.", "weights": [0.0, 0.0, 0.0, 1.0]},
            {"id": "q26", "text": "You prefer to gather more information before deciding.", "weights": [0.0, 0.0, 0.0, -1.0]},
            {"id": "q27", "text": "You are energized by external activities and people.", "weights": [1.0, 0.0, 0.0, 0.0]},
            {"id": "q28", "text": "You focus on what is real and tangible.", "weights": [0.0, 1.0, 0.0, 0.0]},
            {"id": "q29", "text": "You prioritize logical consistency in your reasoning.", "weights": [0.0, 0.0, 1.0, 0.0]},
            {"id": "q30", "text": "You value closure and completing projects.", "weights": [0.0, 0.0, 0.0, 1.0]},
        ],
    }
