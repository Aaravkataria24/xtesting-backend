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
from selenium.common.exceptions import StaleElementReferenceException

# USERNAMES = ["UseUniversalX", "TABASCOweb3", "vaibhavchellani", "intern", "0xMert_", "cryptolyxe", "blknoiz06", "MustStopMurad", "gianinaskarlett", "frankdegods", "notthreadguy", "_TJRTrades", "0xNairolf", "rajgokal", "lukebelmar", "muststopNlG", "VitalikButerin", "TimBeiko", "mauritsneo", "aashatwt", "param_eth", "yashvikram30", "okaykito", "_soulninja", "theunipcs", "cz_binance", "TheCryptoLark", "JupiterExchange", "weremeow", "SOCKETProtocol", "litocoen", "3orovik", "aeyakovenko", "lrettig", "musalbas", "jon_charb", "avsa", "adamscochran", "koeppelmann", "0xCygaar", "cryptunez", "BullyEsq", "solana", "phantom", "ethereum", "SuhailKakar", "IshitaaPandey", "ri5hitripathi"]

# Search Queries = ["chain%20abstraction", "interop", "rollup", "solana", "trenches", "multi-chain", "dApp", "onchain", "web3", "defi", "nft", "gamefi", "socialfi", "dao", "wallet", "staking", "bridging", "L2"]

USERNAME = "HeetTike"
MAX_TWEETS = 300
default_follower_count = 9701  # Set this to the user's follower count before running
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

def get_follower_count(driver):
    try:
        # Use a robust XPath for follower count
        elem = driver.find_element(By.XPATH, '//span[contains(@class, "css-1jxf684") and contains(@class, "r-bcqeeo")]')
        return parse_number(elem.text)
    except Exception as e:
        print(f"Could not get follower count: {e}")
        return 0

def extract_tweet_metadata(tweet):
    # First check if it's a repost - if so, skip immediately
    try:
        if tweet.find_elements(By.XPATH, './/span[contains(text(), "reposted")]'):
            print("[DEBUG] Skipping repost")
            return None
    except:
        pass

    # Click all 'Show more' buttons inside the tweet before extracting text
    show_more_clicked = False
    try:
        show_more_buttons = tweet.find_elements(By.XPATH, ".//span[contains(text(), 'Show more')]/ancestor::button")
        for btn in show_more_buttons:
            try:
                btn.click()
                show_more_clicked = True
                time.sleep(0.2)
            except:
                continue
    except:
        pass

    try:
        # Content (try to get early for debug)
        try:
            content_elem = tweet.find_element(By.XPATH, './/div[@data-testid="tweetText"]')
            if show_more_clicked:
                # If show more was clicked, join text nodes before and after show more
                # Get all direct children (spans, a, etc.)
                children = content_elem.find_elements(By.XPATH, './*')
                lines = []
                buffer = ''
                for child in children:
                    text = child.text
                    if text.strip():
                        if buffer:
                            buffer += text
                            lines.append(buffer)
                            buffer = ''
                        else:
                            lines.append(text)
                    else:
                        # If this is a break (empty), treat as line break
                        if buffer:
                            lines.append(buffer)
                            buffer = ''
                if buffer:
                    lines.append(buffer)
                content = '\n'.join([l.strip() for l in lines if l.strip()])
            else:
                content = content_elem.text.strip()
        except:
            content_elem = tweet.find_element(By.XPATH, './/div[@lang]')
            content = content_elem.text.strip()
    except:
        content = "<no content>"

    # Check if it's quoting another tweet (robust)
    is_quoting = 'yes' if tweet.find_elements(By.XPATH, './/div[starts-with(@id, "id__")][.//div[@data-testid="tweetText"]]') else 'no'

    # Poll detection
    has_poll = 'yes' if tweet.find_elements(By.XPATH, '//*[@id="id__v5v666jccn"]') else 'no'

    print(f"[DEBUG] Processing tweet: {content}")
    try:
        # Engagement
        replies, retweets, likes = extract_engagement(tweet)
        # Date/time
        try:
            timestamp_elem = tweet.find_element(By.XPATH, './/time')
            timestamp = timestamp_elem.get_attribute('datetime')
            date_posted = timestamp.split('T')[0]
            time_posted = timestamp.split('T')[1].split('.')[0]
        except:
            date_posted = time_posted = None
        # Media (exclude quoted tweet containers)
        quoted_divs = tweet.find_elements(By.XPATH, ".//div[starts-with(@id, 'id__')][.//div[@data-testid='tweetText']]")
        def is_in_quoted(elem):
            parent = elem
            while True:
                try:
                    parent = parent.find_element(By.XPATH, "..")
                    if parent in quoted_divs:
                        return True
                except:
                    break
            return False
        # Images
        all_imgs = tweet.find_elements(By.XPATH, './/img[@alt="Image" and contains(@class, "css-9pa8cd")]')
        main_imgs = [img for img in all_imgs if not is_in_quoted(img)]
        # Videos
        all_videos = tweet.find_elements(By.XPATH, './/div[@data-testid="videoPlayer"]//video')
        main_videos = [vid for vid in all_videos if not is_in_quoted(vid)]
        has_video = 'yes' if main_videos else 'no'
        has_image = 'no' if has_video == 'yes' else ('yes' if main_imgs else 'no')
        # Links/mentions
        has_link = 'yes' if tweet.find_elements(By.XPATH, './/a[contains(@href, "http")]') else 'no'
        has_mention = 'yes' if tweet.find_elements(By.XPATH, './/a[starts-with(text(), "@")]') else 'no'
        has_crypto_mention = 'yes' if tweet.find_elements(By.XPATH, './/a[starts-with(text(), "$")]') else 'no'
        # View count
        view_count_elem = tweet.find_elements(By.XPATH, './/a[contains(@aria-label, "views")]//span[contains(@class, "css-1jxf684")]')
        view_count = parse_number(view_count_elem[0].text) if view_count_elem else 0
        # Length
        length = len(content)
        return {
            "content": content,
            "likes": likes,
            "retweets": retweets,
            "replies": replies,
            "date_posted": date_posted,
            "time_posted": time_posted,
            "has_image": has_image,
            "has_video": has_video,
            "has_link": has_link,
            "has_mention": has_mention,
            "has_crypto_mention": has_crypto_mention,
            "length": length,
            "view_count": view_count,
            "is_quoting": is_quoting,
            "has_poll": has_poll
        }
    except Exception as e:
        print(f"Error extracting tweet metadata: {e}")
        return None

def scroll_and_collect(driver, username, max_tweets):
    global tweet_data
    tweet_ids = set()
    scroll_pause = 4
    retry_stages = [5, 10, 10]
    stale_skipped = 0
    error_skipped = 0

    def load_profile():
        driver.get(f"https://x.com/{username}")
        time.sleep(10)

    load_profile()
    follower_count = default_follower_count  # Use the manually set follower count
    last_height = driver.execute_script("return document.body.scrollHeight")
    previous_texts = set()

    while len(tweet_data) < max_tweets:
        tweets = driver.find_elements(By.XPATH, '//article[@role="article"]')
        print(f"🧵 {username}: {len(tweets)} tweets on screen. Total collected: {len(tweet_data)}")

        new_texts = set()
        for tweet_index in range(len(tweets)):
            retry = False
            try:
                tweet = tweets[tweet_index]
                tweet.location_once_scrolled_into_view
                time.sleep(random.uniform(0.2, 0.5))  # Reduce sleep
                meta = extract_tweet_metadata(tweet)
                if meta is None:
                    continue
                content = meta["content"]
                tweet_id = hash(content)
                if tweet_id not in tweet_ids and content:
                    tweet_data.append({
                        "username": username,
                        "follower_count": follower_count,
                        **meta
                    })
                    tweet_ids.add(tweet_id)
                    new_texts.add(content)
            except StaleElementReferenceException:
                # Try to re-fetch the tweet element and retry once
                try:
                    tweets = driver.find_elements(By.XPATH, '//article[@role="article"]')
                    tweet = tweets[tweet_index]
                    tweet.location_once_scrolled_into_view
                    time.sleep(random.uniform(0.2, 0.5))
                    meta = extract_tweet_metadata(tweet)
                    if meta is None:
                        continue
                    content = meta["content"]
                    tweet_id = hash(content)
                    if tweet_id not in tweet_ids and content:
                        tweet_data.append({
                            "username": username,
                            "follower_count": follower_count,
                            **meta
                        })
                        tweet_ids.add(tweet_id)
                        new_texts.add(content)
                    continue
                except StaleElementReferenceException:
                    print("Stale element, skipping tweet after retry.")
                    stale_skipped += 1
                    continue
                except Exception as e:
                    print(f"Error processing tweet after retry: {e}")
                    error_skipped += 1
                    continue
            except Exception as e:
                print(f"Error processing tweet: {e}")
                error_skipped += 1
                continue

        print(f"Skipped {stale_skipped} tweets due to staleness, {error_skipped} due to other errors so far.")

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
    print(f"Total skipped due to staleness: {stale_skipped}, due to other errors: {error_skipped}")
    return tweet_data[:max_tweets]

def handle_exit(signal_received, frame):
    print("\n🔌 Ctrl+C detected. Saving collected tweets before exit...")
    df = pd.DataFrame(tweet_data)
    df.to_csv(f"tweets_user_{USERNAME}.csv", index=False)
    print(f"💾 Saved {len(df)} tweets to 'tweets_user_{USERNAME}.csv'")
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
    df.to_csv(f"tweets_user_{USERNAME}.csv", index=False)
    print(f"\n✅ Done. Saved {len(df)} tweets to 'tweets_user_{USERNAME}.csv'")
    driver.quit()

if __name__ == "__main__":
    main()