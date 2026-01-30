

from ngec.actors.agent_matcher import AgentMatcher


def test_agent_matcher():
    agent_matcher = AgentMatcher()

    res = agent_matcher.trf_agent_match("Syrian Military", "SYR")
    res = agent_matcher.short_text_to_agent("Syrian Military")