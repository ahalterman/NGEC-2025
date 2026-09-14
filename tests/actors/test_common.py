
from ngec.actors.actor_resolution import CountryDetector


def test_country_detector():
    cd = CountryDetector()
    res = cd.search_nat("There were also 5 Americans in the village.")
    assert res == ('USA', 'There were also 5 in the village.')


def test_country_detector_no_country():
    cd = CountryDetector()
    res = cd.search_nat("This text has no country mentioned.")
    assert res == (None, 'This text has no country mentioned.')

def test_country_detector_strips_possessive_mid_span():
    """The possessive goes with the country, wherever the country sits.

    Only a leading "'s" used to be stripped, so "southern Mexico's Zapatista
    rebel group" came back as "southern 's Zapatista rebel group" and was
    searched against Wikipedia in that form.
    """
    cd = CountryDetector()
    assert cd.search_nat("Mexico's Zapatista rebel group") == ("MEX", "Zapatista rebel group")
    assert cd.search_nat("southern Mexico's Zapatista rebel group") == ("MEX", "southern Zapatista rebel group")
