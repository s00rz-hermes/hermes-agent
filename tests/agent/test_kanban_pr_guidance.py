from agent.prompt_builder import KANBAN_GUIDANCE


def test_kanban_guidance_defines_single_pr_owner_and_exact_head_review():
    """Code swarms converge on one draft PR and an exact-head reviewer."""
    assert "one PR-owning card" in KANBAN_GUIDANCE
    assert "first meaningful checkpoint" in KANBAN_GUIDANCE
    assert "draft PR" in KANBAN_GUIDANCE
    assert "exact current head SHA" in KANBAN_GUIDANCE
    assert "same PR branch" in KANBAN_GUIDANCE
    assert "mark the draft PR ready" in KANBAN_GUIDANCE
    assert "independently mergeable" in KANBAN_GUIDANCE
