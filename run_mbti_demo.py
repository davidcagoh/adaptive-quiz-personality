#!/usr/bin/env python3
"""
Interactive MBTI adaptive questionnaire demo.

Uses AdaptiveEngine with VarianceSelection and VarianceThresholdStopping.
"""

from adaptive_quiz.core import AdaptiveEngine, VarianceSelection, VarianceThresholdStopping
from adaptive_quiz.domains.mbti import load_mbti_schema, mbti_from_posterior


def main() -> None:
    """Run interactive MBTI questionnaire."""
    schema = load_mbti_schema()
    question_lookup = {q["id"]: q for q in schema["questions"]}

    engine = AdaptiveEngine(
        schema=schema,
        selection_strategy=VarianceSelection(),
        stopping_rule=VarianceThresholdStopping(variance_threshold=0.2),
    )

    print("=" * 60)
    print("MBTI Adaptive Questionnaire")
    print("=" * 60)
    print("\nRate each statement from -1.0 (strongly disagree) to +1.0 (strongly agree).")
    print("You can use values like -0.5, 0.0, 0.5, etc.\n")

    question_num = 0
    while not engine.is_complete():
        next_q = engine.get_next_question()
        if next_q is None:
            break

        question_id, _ = next_q
        question = question_lookup[question_id]
        question_num += 1

        print(f"\nQuestion {question_num}: {question['text']}")
        while True:
            try:
                response = float(input("Your response (-1.0 to +1.0): "))
                if -1.0 <= response <= 1.0:
                    break
                print("Please enter a value between -1.0 and +1.0.")
            except ValueError:
                print("Please enter a valid number.")

        # Simulate response time (1.0 second default)
        response_time = 1.0
        engine.submit_answer(question_id, response, response_time)

    # Final results
    state = engine.get_state()
    result = mbti_from_posterior(state.mu)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"\nPosterior mean (mu): {state.mu}")
    print(f"\nAxis scores:")
    for axis, value in result["axes"].items():
        print(f"  {axis}: {value:.3f}")
    print(f"\nMBTI Type: {result['type']}")
    print(f"\nQuestions asked: {state.num_questions}")
    print("=" * 60)


if __name__ == "__main__":
    main()
