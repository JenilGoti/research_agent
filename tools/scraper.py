from langchain_core.tools import tool
import requests
from bs4 import BeautifulSoup

@tool
def scrape_url(url: str) -> str:
    """
    Scrapes a webpage and returns clean text content.
    """

    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers, timeout=10)
        html = response.text

        soup = BeautifulSoup(html, "lxml")

        # remove junk
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")

        lines = [line.strip() for line in text.splitlines()]
        cleaned = "\n".join([l for l in lines if len(l) > 30])

        return cleaned[:8000]

    except Exception as e:
        return f"Error scraping {url}: {str(e)}"