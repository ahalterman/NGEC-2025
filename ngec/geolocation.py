from mordecai3 import Geoparser
from rich.progress import track
import time
import jsonlines
import pandas as pd
import os
import logging

from importlib import resources


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def country_name_dict(base_path=None):
    if base_path is None:
        file = str(resources.files('ngec').joinpath("assets", "countries.csv"))
    else:
        file = os.path.join(base_path, "countries.csv")
    countries = pd.read_csv(file)
    country_name_dict = {i:j for i, j in zip(countries['CCA3'], countries['Name'])}
    country_name_dict.update({"": ""})
    country_name_dict.update({"IGO": "Intergovernmental Organization"})
    return country_name_dict


class GeolocationModel:
    def __init__(self,
                geo_model=None,
                nlp=None,
                base_path=None,
                geo_path=None,
                es_client=None,
                save_intermediate=False,
                quiet=False):
        """
        Wrapper around the mordecai3 geoparser.

        Parameters
        ----------
        geo_model: path to a mordecai3 model file. If None (default), mordecai3
            uses the model shipped inside the mordecai3 package.
        nlp: a loaded spaCy model with the "token_tensors" pipe (i.e. the one
            returned by ngec.utilities.load_nlp). Passing the pipeline's own
            spaCy model avoids mordecai3 loading a second copy of
            en_core_web_trf.
        base_path: directory holding countries.csv. If None (default), the copy
            in ngec/assets/ is used.
        geo_path: directory holding mordecai3's geo assets. If None (default),
            mordecai3 uses the assets shipped inside the package.
        es_client: an Elasticsearch client. If None, mordecai3 connects to
            localhost:9200 itself.
        """
        self.geo = Geoparser(model_path=geo_model,
                            geo_asset_path=geo_path,
                            nlp=nlp,
                            es_client=es_client,
                            trim=True,
                            debug=False)
        self.quiet = quiet
        self.save_intermediate = save_intermediate
        self.iso_to_name = country_name_dict(base_path)


    def process(self, story_list, doc_list):
        """
        Wrap the Mordecai3 geoparser function.

        Parameters
        --------
        story_list: list of story dicts. See example
        doc_list: list of spaCy docs
        
        Example
        ------
        event = {'id': '20190801-2227-8b13212ac6f6', 
                'date': '2019-08-01', 
                'event_type': ['SANCTION', 'PROTEST'], 
                'event_mode': [], 
                'event_text': 'The Liberal Party, the largest opposition in Paraguay, announced in the evening of Wednesday the decision to submit an application of impeachment against the president of the country, Mario Abdo Benítez, and vice-president Hugo Velázquez, by polemical agreement with Brazil on the purchase of energy produced in Itaipu. According to the president of the Liberal Party, Efraín Alegre, the opposition also come tomorrow with penal action against all those involved in the negotiations of the agreement with Brazil, signed on confidentiality in May and criticized for being detrimental to the interests of the country. The Liberal Party has the support of the front Guasú, Senator and former President Fernando Lugo, he himself target of an impeachment, decided in less than 24 hours, in June 2012. According to legend, the reasons for the opening of the proceedings against Abdo Benítez are bad performance of functions, betrayal of the homeland and trafficking of influence. Alegre also announced the convocation of demonstrations throughout the country on Friday. ', 
                'story_id': 'EFESP00020190801ef8100001:50066618', 
                'publisher': 'translateme2-pt', 
                'headline': '\nOposição confirma que pedirá impeachment de presidente do Paraguai; PARAGUAI GOVERNO (Pauta)\n', 
                'pub_date': '2019-08-01', 'contexts': ['corruption'], 
                'version': 'NGEC_coder-Vers001-b1-Run-001', 
                'attributes': {'ACTOR': {'text': 'Mario Abdo Benítez', 'score': 0.1976235955953598}, 
                                'RECIP': {'text': 'Fernando Lugo', 'score': 0.10433810204267502}, 
                                'LOC': {'text': 'Paraguay', 'score': 0.24138706922531128}}}
        gp.process([event])
        """
        if len(doc_list) != len(story_list):
            raise ValueError(f"story_list length does not match spaCy doc list len: {len(story_list)} vs. {len(doc_list)}.")

        for n, story in track(enumerate(story_list), total=len(story_list), description="Geoparsing stories..."):
            doc = doc_list[n]
            res = self.geo.geoparse_doc(doc)
            for r in res['geolocated_ents']:
                try:
                    r['country_name'] = self.iso_to_name[r['country_code3']]
                except KeyError:
                    #logger.warning(f"Missing country code for {r}")
                    r['country_name'] = None
                #if 'placename' not in r.keys(): 
                #    print("'placename' key missing from geolocation results")
                #    #print(r)
                #    continue
                #r['search_placename'] = r['placename']
                #if 'resolved_placename' not in r.keys() and 'name' in r.keys():
                #    r['resolved_placename'] = r['name']
                #    del r['name']
                if 'name' in r.keys():
                    r['resolved_placename'] = r['name']
                    del r['name']
            story['geolocated_ents'] = res['geolocated_ents']


        if self.save_intermediate:
            fn = time.strftime("%Y_%m_%d-%H") + "_geolocation_output.jsonl"
            with jsonlines.open(fn, "w") as f:
                f.write_all(story_list)

        return story_list


