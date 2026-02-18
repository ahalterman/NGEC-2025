

from ngec.actors.agent_matcher import AgentMatcher


def test_agent_matcher():
    agent_matcher = AgentMatcher()

    res = agent_matcher.trf_agent_match("Syrian Military", "SYR")
    res = agent_matcher.short_text_to_agent("Syrian Military")


def test_strip_ents():
    agent_matcher = AgentMatcher()

    res1 = agent_matcher.short_text_to_agent("President Barack Obama", 
                                             strip_ents=False)
    assert res1 is not None
    assert res1['query'] == "President Barack Obama"

    res2 = agent_matcher.short_text_to_agent("President Barack Obama",
                                             strip_ents=True)
    assert res2 is not None
    assert res2['query'] == "President"