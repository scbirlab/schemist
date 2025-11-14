"""Tools for querying PubChem."""

from typing import Dict, Iterable, List, Optional, Union
from time import sleep

from carabiner import print_err
from carabiner.cast import cast
from carabiner.decorators import vectorize
from requests import Response, Session

from .http import api_get

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


@api_get(
    url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{query}/property/Title,InchiKey/json",
    allow_error=True,
)
def _inchikey2pubchem_name_id(
    query, 
    r: Response
) -> List[Dict[str, Union[None, int, str]]]:

    j = r.json()
    query = query.split(",")
    defaults = {"pubchem_name": None, "pubchem_id": None}
    if "Fault" in j or r.status_code == 404 or r.status_code in _OVERLOAD_CODES:
        return [defaults for _ in range(len(query))]

    compounds = j["PropertyTable"]["Properties"]
    results = []
    for inchikey in query:
        this_result = [item for item in compounds if item["InChIKey"] == inchikey]

        if len(this_result) > 0:
            this_result = this_result[0]
            name = this_result.get('Title')
            results.append({
                "pubchem_name": name.casefold() if name else None, 
                "pubchem_id": this_result.get('CID'),
            })
        else:
            results.append(defaults)
    return results


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
