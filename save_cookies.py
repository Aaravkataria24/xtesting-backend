from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import pickle

def save_cookies():
    options = Options()
    # Not headless, let you log in manually
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.get("https://twitter.com/login")

    print("Log in manually, then press ENTER here to save cookies")
    input()  # Wait for manual login
    pickle.dump(driver.get_cookies(), open("twitter_cookies.pkl", "wb"))
    driver.quit()

save_cookies()
