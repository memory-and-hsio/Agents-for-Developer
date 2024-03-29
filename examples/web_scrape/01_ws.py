# web scraper
# from https://github.com/jasonrobwebster/langchain-webscraper-demo/blob/master/scrape.py

import argparse
import requests
import os
from urllib.parse import urlparse
from collections import defaultdict
from bs4 import BeautifulSoup
import json



def cleanUrl(url: str):
    return url.replace("https://", "").replace("http://", "").replace("/", "-").replace("?", "_").replace("&", "_").replace("=", "_").replace(".", "_")


def get_response_and_save(url: str):
    response = requests.get(url)

    # create a temp folder
    if not os.path.exists("./scrape"):
        os.mkdir("./scrape")

    # save to file
    parsedUrl = cleanUrl(url)
    with open("./scrape/" + parsedUrl + ".html", "wb") as f:
        f.write(response.content)
    
    return response

def scrape_links(
    scheme: str,
    origin: str,
    path: str,
    depth=3,
    sitemap: dict = defaultdict(lambda: ""),
):
    siteUrl = scheme + "://" + origin + path
    cleanedUrl = cleanUrl(siteUrl)

    if depth < 0:
        return
    if sitemap[cleanedUrl] != "":
        return

    sitemap[cleanedUrl] = siteUrl
    response = get_response_and_save(siteUrl)
    soup = BeautifulSoup(response.content, "html.parser")
    links = soup.find_all("a")

    for link in links:
        href = urlparse(link.get("href"))
        if (href.netloc != origin and href.netloc != "") or (
            href.scheme != "" and href.scheme != "https"
        ):
            continue
        scrape_links(
            href.scheme or "https",
            href.netloc or origin,
            href.path,
            depth=depth - 1,
            sitemap=sitemap,
        )
    return sitemap


# example.  
# python ws.py --site https://folsomtimes.com/ --depth 2
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=str, required=True)
    parser.add_argument("--depth", type=int, default=3)

    args = parser.parse_args()
    url = urlparse(args.site)

    sitemap = scrape_links(url.scheme, url.netloc, url.path, depth=args.depth)

    with open("./scrape/sitemap.json", "w") as f:
        f.write(json.dumps(sitemap))
    
    print("Sitemap saved to sitemap.json")


