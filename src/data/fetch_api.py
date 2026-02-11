# src/data/fetch_api.py
import requests
import time
from typing import List, Dict

ANILIST_API = "https://graphql.anilist.co"

def query_anilist_media(page: int = 1, per_page: int = 50, search: str = None):
    query = '''
    query ($page: Int, $perPage: Int, $search: String) {
      Page(page: $page, perPage: $perPage) {
        media(search: $search, type: ANIME) {
          id
          idMal
          title { romaji english native }
          description(asHtml: false)
          genres
          tags { name rank }
          averageScore
          popularity
          startDate { year month day }
          siteUrl
          coverImage { large medium color }
        }
      }
    }
    '''
    variables = {"page": page, "perPage": per_page, "search": search}
    r = requests.post(ANILIST_API, json={'query': query, 'variables': variables})
    r.raise_for_status()
    return r.json()

def fetch_all_media(max_pages=5, per_page=50, sleep_sec=1.0) -> List[Dict]:
    results = []
    for p in range(1, max_pages+1):
        resp = query_anilist_media(page=p, per_page=per_page)
        media = resp.get('data', {}).get('Page', {}).get('media', [])
        if not media:
            break
        results.extend(media)
        time.sleep(sleep_sec)
    return results

if __name__ == "__main__":
    m = fetch_all_media(1,50)
    print('fetched', len(m))
