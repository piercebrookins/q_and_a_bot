from slack_db_bot.evaluation import case_passed, grade_text


def test_grade_accepts_plural_and_phrase_equivalence() -> None:
    grade = grade_text(
        "We will add a taxonomy translation layer with alias mapping and coerce values to integer.",
        ["taxonomy mapping layer", "alias mappings", "integers"],
    )
    assert grade["all_required_terms"]


def test_grade_accepts_written_date_but_requires_exact_command() -> None:
    text = "The window is March 23, 2026. Run orchestrator rollback --target ruleset=abc."
    grade = grade_text(text, ["2026-03-23", "orchestrator rollback --target ruleset=<prior_sha>"])
    assert grade["required_term_recall"] == 0.5
    assert grade["missing_terms"] == ["orchestrator rollback --target ruleset=<prior_sha>"]


def test_grade_accepts_approval_path_latency() -> None:
    grade = grade_text("Tracing reports approval-path latency.", ["approval latency"])
    assert grade["all_required_terms"]


def test_quality_gate_rejects_keyword_complete_but_semantically_wrong_answer() -> None:
    result = {
        "all_required_terms": True,
        "abstained": False,
        "within_action_budget": True,
        "within_latency_budget": True,
        "semantic_grade": {
            "correct": False,
            "complete": True,
            "grounded": True,
            "associations_correct": False,
            "contradictions": ["The answer assigns the milestone to the wrong customer."],
            "unsupported_claims": [],
        },
    }
    assert not case_passed(result)
