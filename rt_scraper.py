"""Selenium parser for collecting current Rotten Tomatoes critic reviews."""

import argparse
import json
import re
import time
from pathlib import Path

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def build_browser(headless=True):
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1440,1200")
    return webdriver.Chrome(options=options)


def scrape_movie(browser, movie, page_num=1):
    """Return critic reviews from one Rotten Tomatoes movie slug."""
    wait = WebDriverWait(browser, 20)
    url = f"https://www.rottentomatoes.com/m/{movie}/reviews/all-critics"
    browser.get(url)

    try:
        button = WebDriverWait(browser, 4).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "//button[contains(normalize-space(.), 'Continue')]"
                    " | //rt-button[contains(normalize-space(.), 'Continue')]",
                )
            )
        )
        browser.execute_script("arguments[0].click();", button)
    except TimeoutException:
        pass

    review_xpath = "//*[normalize-space(.)='Go to Full Review']"
    wait.until(EC.presence_of_element_located((By.XPATH, review_xpath)))

    for _ in range(1, page_num):
        old_count = len(browser.find_elements(By.XPATH, review_xpath))
        try:
            load_more = wait.until(
                EC.element_to_be_clickable(
                    (
                        By.XPATH,
                        "//button[normalize-space(.)='Load More']"
                        " | //rt-button[normalize-space(.)='Load More']"
                        " | //*[@data-qa='load-more-btn']",
                    )
                )
            )
            browser.execute_script(
                "arguments[0].scrollIntoView({block: 'center'});", load_more
            )
            browser.execute_script("arguments[0].click();", load_more)
            wait.until(
                lambda driver: len(driver.find_elements(By.XPATH, review_xpath))
                > old_count
            )
            time.sleep(0.5)
        except TimeoutException:
            break

    reviews = []
    seen = set()
    for link in browser.find_elements(By.XPATH, review_xpath):
        card = link
        for _ in range(12):
            card = card.find_element(By.XPATH, "..")
            card_text = card.text.strip()
            if (
                re.search(r"[A-Z][a-z]{2} \d{1,2}, \d{4}", card_text)
                and "Go to Full Review" in card_text
                and len(card_text.splitlines()) >= 4
            ):
                break

        lines = [line.strip() for line in card.text.splitlines() if line.strip()]
        date = None
        date_index = None
        for index, line in enumerate(lines):
            if re.fullmatch(r"[A-Z][a-z]{2} \d{1,2}, \d{4}", line):
                date = line
                date_index = index
                break

        rating = None
        rating_elements = card.find_elements(
            By.CSS_SELECTOR,
            "score-icon-critics, score-icon-critic, score-icon-critic-deprecated, "
            "[sentiment], [data-qa='review-rating'], img",
        )
        for element in rating_elements:
            info = " ".join(
                [
                    element.get_attribute("sentiment") or "",
                    element.get_attribute("state") or "",
                    element.get_attribute("alt") or "",
                    element.get_attribute("src") or "",
                    element.get_attribute("class") or "",
                ]
            ).lower()
            if "rotten" in info:
                rating = "rotten"
                break
            if "fresh" in info or "tomato" in info:
                rating = "fresh"
                break

        text = None
        if date_index is not None:
            text_lines = []
            for line in lines[date_index + 1 :]:
                if "Go to Full Review" in line:
                    break
                if re.fullmatch(r"\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?", line):
                    continue
                text_lines.append(line)
            text = " ".join(text_lines) or None

        review = {
            "movie_id": movie,
            "critic_name": lines[0] if date_index and date_index > 0 else None,
            "rating": rating,
            "source": lines[1] if date_index and date_index > 1 else None,
            "text": text,
            "date": date,
        }
        key = tuple(review.values())
        if key not in seen:
            seen.add(key)
            reviews.append(review)
    return reviews


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("movie")
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--visible", action="store_true")
    args = parser.parse_args()

    browser = build_browser(headless=not args.visible)
    try:
        reviews = scrape_movie(browser, args.movie, args.pages)
    finally:
        browser.quit()

    output = Path(f"ishfaq_shah_{args.movie}.json")
    output.write_text(json.dumps(reviews, indent=2, ensure_ascii=False))
    print(f"Saved {len(reviews)} reviews to {output}")


if __name__ == "__main__":
    main()
