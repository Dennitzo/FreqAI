"""Record local SearXNG research and primary references for unpaired text fields."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/information_corpus/research"
QUERIES = [
    "FNet Fourier transforms unparameterized token mixing language model training paper",
    "spectral algorithm learning hidden Markov models observable moments Hsu Kakade Zhang",
    "holographic reduced representations circular convolution associative memory Plate 1995",
    "Shannon mathematical theory communication word approximation transition probabilities 1948",
]
PRIMARY = {
    "fnet.html": "https://aclanthology.org/2022.naacl-main.319/",
    "spectral-hmm.html": "https://arxiv.org/abs/0811.4413",
    "associative-memory.html": "https://proceedings.mlr.press/v48/danihelka16.html",
    "numpy-fft.html": "https://numpy.org/doc/stable/reference/routines.fft.html",
}


def search(query):
    body = json.dumps({"query": query, "maximumResults": 8, "language": "de-DE"}).encode()
    req = urllib.request.Request("http://127.0.0.1:8080/v1/research/web", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as response:
        value = json.load(response)
    assert value.get("provider") == "searxng" and value.get("isFallback") is False, value.keys()
    return {"query": query, "response": value}


def primary(item):
    name, url = item
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "FreqAI research/0.1"}), timeout=60) as response:
            body, status = response.read(), response.status
        (OUT / name).write_bytes(body)
        return {"url": url, "file": name, "status": status, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    except Exception as error:
        return {"url": url, "error": str(error)}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        searches = list(pool.map(search, QUERIES))
        sources = list(pool.map(primary, PRIMARY.items()))
    (OUT / "algorithm_research.json").write_text(json.dumps({"searches": searches, "sources": sources}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"local_verified_searches": len(searches), "primary_sources": sources}), flush=True)
