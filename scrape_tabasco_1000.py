import time
import random
import pickle
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import signal
import sys

# USERNAMES = ["TABASCOweb3", "vaibhavchellani", "intern", "0xMert_", "cryptolyxe", "blknoiz06", "MustStopMurad", "gianinaskarlett", "frankdegods", "notthreadguy", "_TJRTrades", "0xNairolf", "rajgokal", "lukebelmar", "muststopnigg", "VitalikButerin", "TimBeiko", "gavofyork", "cz_binance", "TheCryptoLark", "JupiterExchange", "weremeow", "SOCKETProtocol", "litocoen", "3orovik", "aeyakovenko", "lrettig", "musalbas", "jon_charb", "avsa", "adamscochran", "koeppelmann", "0xCygaar", "cryptunez", "BullyEsq", "solana", "phantom", "ethereum"]

# Search Queries = ["chain%20abstraction", "interop", "rollup", "solana", "trenches", "multi-chain", "dApp", "onchain", "web3", "defi", "nft", "gamefi", "socialfi", "dao", "wallet", "staking", "bridging", "L2"]

USERNAME = "L2"
MAX_TWEETS = 1000
tweet_data = []  # Global list to store tweets for interrupt handler

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

def parse_number(text):
    if not text:
        return 0
    text = text.upper().replace(',', '').strip()
    try:
        if 'K' in text:
            return int(float(text.replace('K', '')) * 1_000)
        elif 'M' in text:
            return int(float(text.replace('M', '')) * 1_000_000)
        return int(text)
    except:
        return 0

def extract_engagement(tweet):
    try:
        container = tweet.find_element(By.XPATH, './/div[@role="group"]')
        buttons = container.find_elements(By.XPATH, './div')
        replies = retweets = likes = 0
        if len(buttons) >= 3:
            for idx, label in zip([0, 1, 2], ["Replies", "Retweets", "Likes"]):
                try:
                    count_span = buttons[idx].find_element(By.XPATH, './/span/span/span')
                    count = parse_number(count_span.text.strip())
                    if label == "Replies": replies = count
                    elif label == "Retweets": retweets = count
                    elif label == "Likes": likes = count
                except:
                    continue
        return replies, retweets, likes
    except:
        return 0, 0, 0

def get_visible_tweet_texts(driver):
    tweets = driver.find_elements(By.XPATH, '//article[@role="article"]')
    texts = set()
    for tweet in tweets:
        try:
            try:
                content_elem = tweet.find_element(By.XPATH, './/div[@data-testid="tweetText"]')
            except:
                content_elem = tweet.find_element(By.XPATH, './/div[@lang]')
            content = content_elem.text.strip()
            if content:
                texts.add(content)
        except:
            continue
    return texts

def click_retry(driver, attempts, sleep_time, previous_texts):
    for attempt in range(1, attempts + 1):
        current_texts = get_visible_tweet_texts(driver)
        if not current_texts.issubset(previous_texts):
            print("✅ New tweets detected during retry. Skipping retries.")
            return False

        print(f"🔁 Retry attempt {attempt}/{attempts}")
        try:
            retry_btn = driver.find_element(By.XPATH, '//span[contains(text(), "Try again") or contains(text(), "Retry")]')
            retry_btn.click()
            time.sleep(sleep_time + attempt * 2)
        except:
            time.sleep(sleep_time + attempt)
    return True

def scroll_and_collect(driver, username, max_tweets):
    global tweet_data
    tweet_ids = set()
    scroll_pause = 4
    retry_stages = [5, 10, 10]

    def load_profile():
        driver.get(f"https://x.com/search?q={username}&src=typed_query")
        time.sleep(10)

    load_profile()
    last_height = driver.execute_script("return document.body.scrollHeight")
    previous_texts = set()

    while len(tweet_data) < max_tweets:
        tweets = driver.find_elements(By.XPATH, '//article[@role="article"]')
        print(f"🧵 {username}: {len(tweets)} tweets on screen. Total collected: {len(tweet_data)}")

        new_texts = set()
        for tweet in tweets:
            try:
                tweet.location_once_scrolled_into_view
                time.sleep(random.uniform(0.5, 1.0))
                try:
                    content_elem = tweet.find_element(By.XPATH, './/div[@data-testid="tweetText"]')
                except:
                    content_elem = tweet.find_element(By.XPATH, './/div[@lang]')
                content = content_elem.text.strip()
                tweet_id = hash(content)
                if tweet_id not in tweet_ids and content:
                    replies, retweets, likes = extract_engagement(tweet)
                    tweet_data.append({
                        "username": "search query ethereum",
                        "content": content,
                        "likes": likes,
                        "retweets": retweets,
                        "replies": replies
                    })
                    tweet_ids.add(tweet_id)
                    new_texts.add(content)
            except:
                continue

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(random.uniform(scroll_pause, scroll_pause + 2))

        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height and not new_texts:
            print(f"⚠️ {username}: Stuck while scrolling. Entering retry phases...")
            for stage_index, attempts in enumerate(retry_stages):
                print(f"🚨 Retry phase {stage_index + 1}")
                retry_success = click_retry(driver, attempts, 10, previous_texts)
                current_texts = get_visible_tweet_texts(driver)
                if not current_texts.issubset(previous_texts):
                    print("✅ New tweets appeared after retry click.")
                    break
                elif stage_index < len(retry_stages) - 1:
                    print("😴 Waiting 3 minutes before next retry phase...")
                    time.sleep(180)
            else:
                print(f"⛔ All retries failed. Ending scrape for @{username}.")
                break
        else:
            previous_texts.update(new_texts)
            last_height = new_height

    print(f"✅ Final count for @{username}: {len(tweet_data)} tweets")
    return tweet_data[:max_tweets]

def handle_exit(signal_received, frame):
    print("\n🔌 Ctrl+C detected. Saving collected tweets before exit...")
    df = pd.DataFrame(tweet_data)
    df.to_csv(f"tweets_search_query_{USERNAME}.csv", index=False)
    print(f"💾 Saved {len(df)} tweets to 'tweets_search_query_{USERNAME}.csv'")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    driver = setup_driver()
    driver.get("https://twitter.com")
    time.sleep(4)

    cookies = pickle.load(open("twitter_cookies.pkl", "rb"))
    for cookie in cookies:
        driver.add_cookie(cookie)

    print(f"\n🔍 Scraping @{USERNAME}...")
    scroll_and_collect(driver, USERNAME, MAX_TWEETS)

    df = pd.DataFrame(tweet_data)
    df.to_csv(f"tweets_search_query_{USERNAME}.csv", index=False)
    print(f"\n✅ Done. Saved {len(df)} tweets to 'tweets_search_query_{USERNAME}.csv'")
    driver.quit()

if __name__ == "__main__":
    main()
