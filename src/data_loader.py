
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import time

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_DIR / "publications.csv"

CATEGORIES = ["cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.RO"]

MAX_RESULTS = 1000

BASE_URL = "https://export.arxiv.org/api/query"

# --------------------------------------------------
# 2. Download publications from arXiv
# --------------------------------------------------

def download_publications():

    publications = []

    query = " OR ".join(
        [f"cat:{category}" for category in CATEGORIES]
    )

    params = {
        "search_query": f"({query})",
        "start": 0,
        "max_results": MAX_RESULTS,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }

    print("Downloading scientific publications...")

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    # Parse the XML response
    root = ET.fromstring(response.content)

    namespace = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom"
    }

    for entry in root.findall("atom:entry", namespace):

        paper_id = entry.find(
            "atom:id", namespace
        ).text

        title = entry.find(
            "atom:title", namespace
        ).text

        abstract = entry.find(
            "atom:summary", namespace
        ).text

        categories = [
            category.attrib["term"]
            for category in entry.findall(
                "atom:category", namespace
            )
        ]

        publications.append({
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "categories": ", ".join(categories)
        })

    return publications


# --------------------------------------------------
# 3. Save dataset
# --------------------------------------------------

if __name__ == "__main__":

    papers = download_publications()

    df = pd.DataFrame(papers)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nDataset successfully saved!")
    print("Number of publications:", len(df))
    print("File:", OUTPUT_FILE)

    print("\nFirst five publications:")
    print(df.head())

