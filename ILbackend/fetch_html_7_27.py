FIRECRAWL_LOCK = threading.Lock()        # <‑‑ add this near your imports


# def fetch_html(url: str) -> str:
#     """
#     Fetch *url* with Firecrawl, retrying on 429/timeout.  A global mutex
#     (`FIRECRAWL_LOCK`) guarantees that **only one Firecrawl scrape is
#     active at any moment**, preventing overlapping sessions.
#     """
#     with FIRECRAWL_LOCK:                 # <‑‑ critical section
#         for attempt in range(1, MAX_RETRIES + 1):
#             logger.info(f"[FIRECRAWL] Scraping {url} (try {attempt}/{MAX_RETRIES})")
#             try:
#                 r = fc_client.scrape_url(url=url, formats=["html"])
#                 return r.html or ""
#             except Exception as e:
#                 # 429 = rate‑limit, any timeout counts as transient
#                 transient = any(tok in str(e) for tok in ("429", "Timeout"))
#                 if transient and attempt < MAX_RETRIES:
#                     logger.warning(
#                         f"[FIRECRAWL WAIT] transient error: {e}. Sleeping {RATE_LIMIT_SLEEP}s."
#                     )
#                     time.sleep(RATE_LIMIT_SLEEP)
#                     continue

#                 logger.error(f"[FIRECRAWL ERROR] {e}")
#                 return ""

#         logger.error(f"[FIRECRAWL ERROR] failed after {MAX_RETRIES} retries")
#         return ""
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.common.exceptions import TimeoutException, WebDriverException

FIRECRAWL_LOCK = threading.Lock()  # Global lock still used
MAX_RETRIES = 3
RATE_LIMIT_SLEEP = 2  # seconds

def fetch_html(url: str) -> str:
    """
    Fetch *url* using headless Selenium with retry logic.
    Uses a global mutex (`FIRECRAWL_LOCK`) to prevent concurrent scrapes.
    """
    options = FirefoxOptions()
    options.headless = True

    with FIRECRAWL_LOCK:
        for attempt in range(1, MAX_RETRIES + 1):
            logger.info(f"[SELENIUM] Scraping {url} (try {attempt}/{MAX_RETRIES})")
            try:
                driver = webdriver.Firefox(options=options)
                driver.set_page_load_timeout(20)  # Timeout in seconds

                driver.get(url)
                html = driver.page_source
                driver.quit()
                return html

            except (TimeoutException, WebDriverException) as e:
                logger.warning(f"[SELENIUM ERROR] transient error: {e}")
                if attempt < MAX_RETRIES:
                    logger.warning(f"[SELENIUM WAIT] Sleeping {RATE_LIMIT_SLEEP}s before retry.")
                    time.sleep(RATE_LIMIT_SLEEP)
                    continue
                else:
                    logger.error(f"[SELENIUM ERROR] failed after {MAX_RETRIES} retries: {e}")
                    return ""

            finally:
                try:
                    driver.quit()
                except Exception:
                    pass  # Ensure we always quit driver if it's open

    return ""