import os
import requests
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langgraph.errors import GraphRecursionError

from cv_agent.data_models import ScrapeResult, ScrapeFailed

from dotenv import load_dotenv
load_dotenv() 

MODEL = os.getenv("SCRAPER_MODEL", "anthropic:claude-haiku-5-5")
MAX_STEPS = 20

SYSTEM_PROMPT = """Get the full text of the job offer at the given URL.
Try fetch_page first. If the text is missing or incomplete, use browse,
clicking cookie banners or "show more" buttons if needed.
Copy the offer text as it appears; don't summarize or invent anything.
If it's not a job offer or you can't get it, return ScrapeFailed."""

def _chrome() -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(options=options)

def _build_tools():
    @tool
    def fetch_page(url: str) -> str:
        """Fetch a page with plain HTTP (fast, no JavaScript) and return its text."""
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            return f"Fetch failed: {e}"

    @tool
    def browse(url: str, click: list[str] | None = None) -> str:
        """Open the page in a real browser (handles JavaScript) and return its text.
        Optionally click elements by visible text first, e.g. ["Akceptuję", "Pokaż więcej"]."""
        try:
            with _chrome() as driver:
                driver.get(url)
                time.sleep(2)                                  # let JS render the content
                for text in click or []:
                    xpath = f"//*[self::button or self::a or @role='button'][contains(normalize-space(.), \"{text}\")]"
                    elements = driver.find_elements(By.XPATH, xpath)
                    if elements:
                        driver.execute_script("arguments[0].click();", elements[0])
                        time.sleep(1)
                return driver.page_source
        except WebDriverException as e:
            return f"Browser failed: {e.msg}"

    return [fetch_page, browse]

def scrape(url: str) -> ScrapeResult:
    """Get the full text of the job offer at `url`."""
    agent = create_agent(
        model=MODEL,
        tools=_build_tools(),
        system_prompt=SYSTEM_PROMPT,
        response_format=ScrapeResult,
    )
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": f"Get the job offer from: {url}"}]},
            {"recursion_limit": MAX_STEPS},
        )
        return result["structured_response"]
    except GraphRecursionError:
        return ScrapeResult(result = ScrapeFailed(reason=f"Agent did not finish within {MAX_STEPS} steps."))


if __name__ == "__main__":
    scraped = scrape("https://www.linkedin.com/jobs/view/software-engineer-backend-at-plaid-4410261275?trk=public_jobs_topcard-title")

    print(scraped)