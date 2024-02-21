"""Tools for querying PubChem."""

from typing import Dict, Iterable, List, Optional, Union
from time import sleep
from xml.etree import ElementTree

from carabiner import print_err
from carabiner.cast import cast
from carabiner.decorators import vectorize
from requests import Response, Session

_PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/property/{get}/{format}"
_CACTUS_URL = "https://cactus.nci.nih.gov/chemical/structure/{inchikey}/{get}"

_OVERLOAD_CODES = {500, 501, 503, 504}


def _url_request(inchikeys: Union[str, Iterable[str]],
                 url: str,
                 session: Optional[Session] = None, 
                 **kwargs) -> Response:

    if session is None:
        session = Session()

    inchikeys = cast(inchikeys, to=list)

    return session.get(url.format(inchikey=','.join(inchikeys), **kwargs))


def _inchikey2pubchem_name_id(inchikeys: Union[str, Iterable[str]], 
                        session: Optional[Session] = None, 
                        counter: int = 0, 
                        max_tries: int = 10,
                        namespace: str = "{http://pubchem.ncbi.nlm.nih.gov/pug_rest}") -> List[Dict[str, Union[None, int, str]]]:

    r = _url_request(inchikeys, url=_PUBCHEM_URL, 
                     session=session, 
                     get="Title,InchiKey", format="XML")

    if r.status_code == 200:

        root = ElementTree.fromstring(r.text)
        compounds = root.iter(f'{namespace}Properties')

        result_dict = dict()
        
        for cmpd in compounds:
            
            cmpd_dict = dict()
            
            for child in cmpd:
                cmpd_dict[child.tag.split(namespace)[1]] = child.text
            
            try:
                inchikey, name, pcid = cmpd_dict['InChIKey'], cmpd_dict['Title'], cmpd_dict['CID']
            except KeyError:
                print(cmpd_dict)
            else:
                result_dict[inchikey] = {'pubchem_name': name.casefold(), 
                                         'pubchem_id': pcid}

        print_err(f'PubChem: Looked up InchiKeys: {",".join(inchikeys)}')
  
        result_list = [result_dict[inchikey] 
                       if inchikey in result_dict 
                       else {'pubchem_name': None, 'pubchem_id': None}
                       for inchikey in inchikeys]

        return result_list

    elif r.status_code in _OVERLOAD_CODES and counter < max_tries:

        sleep(1.)

        return _inchikey2pubchem_name_id(inchikeys, 
                                         session=session, 
                                         counter=counter + 1, 
                                         max_tries=max_tries, 
                                         namespace=namespace)

    else:
        
        print_err(f'PubChem: InchiKey {",".join(inchikeys)} gave status {r.status_code}')
        
        return [{'pubchem_name': None, 'pubchem_id': None} 
                for _ in range(len(inchikeys))]


@vectorize
def _inchikey2cactus_name(inchikeys: str, 
                          session: Optional[Session] = None, 
                          counter: int = 0, 
                          max_tries: int = 10):

    r = _url_request(inchikeys, url=_CACTUS_URL, 
                     session=session, 
                     get="names")

    if r.status_code == 200:

        return r.text.split('\n')[0].casefold()

    elif r.status_code in _OVERLOAD_CODES and counter < max_tries:

        sleep(1.)

        return _inchikey2cactus_name(inchikeys, 
                                     session=session, 
                                     counter=counter + 1, 
                                     max_tries=max_tries)

    else:
        
        print_err(f'Cactus: InchiKey {",".join(inchikeys)} gave status {r.status_code}')
        
        return None

