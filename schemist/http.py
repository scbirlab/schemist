"""Fetching remote data."""

from typing import Any, Callable, Mapping, Optional
from functools import cache, wraps
import os
import time

from carabiner import print_err, pprint_dict
from carabiner.decorators import decorator_with_params
import requests
from requests import Response

from . import app_name, __version__, __author__
from importlib.metadata import metadata


@decorator_with_params
def api_get(
    f: Callable[[str, Response], Any],
    url: str,
    max_tries: int = 3,
    query_key: Optional[str] = None,
    default_params: Optional[Mapping[str, Any]] = None,
    cache_dir: Optional[str] = None,
    allow_error: bool = False
) -> Callable[[Optional[str], Optional[dict]], Any]:
    default_params = default_params or {}
    url0 = url

    headers = {
        "User-Agent": f"{app_name}/{__version__}",
        "From": (
            metadata(app_name)["Author-email"]
            .replace(__author__, "")
            .strip()
        ),
    }


    def api_call(
        query=None, 
        params=None,
        wait: float = .2,
        _try: int = 0,
        *args, **kwargs
    ):
        time.sleep(wait)
        params = default_params | (params or {})
        url = url0
        if query_key is not None and query is not None:
            params = params | {query_key: query}
        elif query is not None:
            if isinstance(query, (list, tuple)):
                query=",".join(query)
            url = url.format(query=query)
        pprint_dict(
            params,
            message=f"Downloading from '{url}' with the following parameters"
        )
        try:
            r = requests.get(url, params=params, headers=headers)
        except requests.exceptions.ConnectionError as e:
            print_err(e)
            next_try = _try + 1
            print_err(f"[INFO] Tried {next_try} / {max_tries} times...", end=" ")
            if next_try < max_tries:
                print_err("")
                return api_call(
                    query=query, 
                    params=params,
                    wait=wait * 2,
                    _try=next_try,
                    *args, **kwargs
                )
            else:
                print_err("stopping!")
                raise e
        else:
            print_err(f"Trying {r.url}... {r.status_code}", end="")
            if not allow_error:
                r.raise_for_status()
            print_err(f"... ok")
            return f(query, r, *args, **kwargs)

    if cache_dir is not None and isinstance(cache_dir, str):
        from joblib import Memory
        mem = Memory(location=os.path.join(cache_dir, "api_calls"), verbose=0)
        api_call = mem.cache(api_call)
        
    return wraps(f)(cache(api_call))
