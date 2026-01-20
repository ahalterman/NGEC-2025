# This test already existsin ngec/tests/test_formatter.py but the whole thing
# chokes when ES is not available during the conftest.py setup. 


from ngec.formatter import resolve_date

def test_resolution():
    event = {"pub_date": "June 20, 2012",
             "attributes": {"date": ["last Sunday"]}
             }
    res = resolve_date(event)
    # this fails in a substantive sense, and it falls back on pub date
    assert "date_resolved" in res
    assert res["date_resolved"]["resolved_date"].strftime("%Y-%m-%d") == "2012-06-20"
    assert "failed to convert relative date" in res["date_resolved"]["reason"]
