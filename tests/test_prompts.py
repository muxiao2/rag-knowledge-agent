from rag_agent.prompts import RAG_PROMPT_TEMPLATE, SYSTEM_PROMPT, format_context


def test_format_context_empty():
    assert "no relevant context" in format_context([])


def test_format_context_numbers_snippets():
    out = format_context(["first", "second"])
    assert "[1] first" in out
    assert "[2] second" in out


def test_prompt_template_includes_parts():
    prompt = RAG_PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT, context="ctx", question="q?"
    )
    assert "ctx" in prompt
    assert "q?" in prompt
    assert "knowledge-base assistant" in prompt
