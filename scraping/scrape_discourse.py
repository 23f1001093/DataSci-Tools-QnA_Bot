import os
import json
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError, Error
from bs4 import BeautifulSoup
import logging
import time # Import time for delays
import traceback # Import traceback for detailed error logging
import re # Import re for regex operations
from tqdm import tqdm # Import tqdm for progress bars

# Setup logging - SET TO DEBUG FOR DETAILED OUTPUT
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === CONFIG ===
BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY_ID = 34 # This is specific to "Tools in Data Science KB"
CATEGORY_JSON_URL = f"{BASE_URL}/c/courses/tds-kb/{CATEGORY_ID}.json"
AUTH_STATE_FILE = "auth.json"
START_DATE = datetime(2025, 1, 1, 0, 0, 0)
END_DATE = datetime(2025, 4, 14, 23, 59, 59) # Inclusive of April 14th
DOWNLOAD_DIR = "downloaded_threads" 

# Ensure download directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
logger.info(f"Ensured '{DOWNLOAD_DIR}' directory exists.")

def parse_date(date_str):
    """Parses a date string from Discourse API into a datetime object."""
    try:
        # Try format with milliseconds and Z
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        # Fallback to format without milliseconds and with Z
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")

def login_and_save_auth(playwright):
    """
    Launches a non-headless browser for manual login and saves the session state.
    """
    logger.info("🔐 No valid auth found or session invalid. Launching browser for manual login...")
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    
    try:
        logger.info(f"Navigating to login page: {BASE_URL}/login")
        page.goto(f"{BASE_URL}/login", timeout=90000) # Increased timeout significantly
        logger.info("🌐 Please log in manually using Google in the browser window that just opened.")
        logger.info("🚨 IMPORTANT: After logging in and seeing your Discourse dashboard (not just the Google login page), wait a few seconds for the page to fully load.")
        logger.info("🚨 THEN, manually close the browser window.")
        
        # This prompt gives the user time to log in and observe the dashboard
        input("Press Enter in THIS terminal after you have successfully logged in and CLOSED THE BROWSER WINDOW...")
        
        # After user closes browser, try to save the state from the context
        context.storage_state(path=AUTH_STATE_FILE)
        logger.info(f"✅ Login state saved successfully to {AUTH_STATE_FILE}. File size: {os.path.getsize(AUTH_STATE_FILE)} bytes.")
    except Exception as e:
        logger.error(f"❌ Error during manual login phase: {e}")
        logger.error(traceback.format_exc())
        logger.error("Please ensure you logged in correctly, closed the browser, and had network access.")
    finally:
        # Ensure browser is closed even if an error occurs
        if browser:
            browser.close()
            logger.debug("Browser closed after manual login attempt.")

def is_authenticated(context):
    """
    Checks if the loaded session is still authenticated by trying to access a protected JSON URL.
    """
    page = context.new_page()
    try:
        logger.debug(f"Checking authentication by navigating to: {CATEGORY_JSON_URL}")
        page.goto(CATEGORY_JSON_URL, timeout=15000) # Increased timeout
        page.wait_for_selector("pre", timeout=10000) # Wait for the pre tag which should contain JSON
        json_content = page.inner_text("pre")
        json_data = json.loads(json_content)
        # Further check: does the JSON look like a valid topic list?
        if "topic_list" in json_data and "topics" in json_data["topic_list"]:
            logger.info("✅ Existing authenticated session appears valid (JSON content received).")
            return True
        else:
            logger.warning("⚠️ Session invalid: Received JSON but it doesn't contain expected 'topic_list'.")
            return False
    except (TimeoutError, json.JSONDecodeError, Error) as e:
        logger.warning(f"⚠️ Session invalid: Could not fetch JSON or parse it. Re-authentication likely required. Error: {e}")
        logger.debug(traceback.format_exc())
        return False
    finally:
        page.close() # Always close the page

def scrape_posts(playwright):
    """
    Scrapes Discourse topics and posts within the defined date range using Playwright.
    """
    logger.info("🔍 Starting scrape using Playwright session...")
    browser = None
    try:
        # Launch browser with the saved authentication state
        browser = playwright.chromium.launch(headless=True) # Run headless for scraping
        context = browser.new_context(storage_state=AUTH_STATE_FILE)
        page = context.new_page()

        # Verify authentication before proceeding
        logger.info("Performing pre-scrape authentication check...")
        if not is_authenticated(context):
            logger.error("❌ Authentication failed during pre-scrape check. Please re-run the script and perform manual login carefully.")
            return

        all_topics = []
        page_num = 0
        
        # Loop through paginated topic lists
        while True:
            paginated_url = f"{CATEGORY_JSON_URL}?page={page_num}"
            logger.info(f"📦 Fetching topic list page {page_num} from {paginated_url}...")
            
            try:
                page.goto(paginated_url, timeout=45000) # Increased timeout
                page.wait_for_selector("pre", timeout=15000) # Wait for the pre tag with JSON content
                data = json.loads(page.inner_text("pre"))
            except (TimeoutError, json.JSONDecodeError, Error) as e:
                logger.warning(f"Could not fetch or parse topic list page {page_num}: {e}. Assuming no more pages or error, breaking loop.")
                logger.debug(traceback.format_exc())
                break # Exit loop if page fails to load or parse

            topics = data.get("topic_list", {}).get("topics", [])
            if not topics:
                logger.info(f"No topics found on page {page_num}. Ending topic list fetching.")
                break # No more topics on this page, or end of list

            all_topics.extend(topics)
            page_num += 1
            time.sleep(1) # Small delay between fetching topic list pages

        logger.info(f"📄 Found {len(all_topics)} total topic summaries. Filtering by date range...")

        os.makedirs(DOWNLOAD_DIR, exist_ok=True) # Ensure directory exists
        saved_posts_count = 0

        # Iterate through all fetched topic summaries and download details if within date range
        for topic_summary in tqdm(all_topics, desc="Downloading topic details"):
            topic_id = topic_summary.get("id")
            created_at_str = topic_summary.get("created_at")
            
            if not topic_id or not created_at_str:
                logger.warning(f"Skipping malformed topic summary: {topic_summary}")
                continue

            try:
                topic_created_at = parse_date(created_at_str)
                
                if START_DATE <= topic_created_at <= END_DATE:
                    topic_slug = topic_summary.get('slug', 'no-slug')
                    topic_url = f"{BASE_URL}/t/{topic_slug}/{topic_id}.json"
                    
                    logger.debug(f"Attempting to fetch details for topic {topic_id} ({topic_url})...")
                    page.goto(topic_url, timeout=45000) # Increased timeout
                    page.wait_for_selector("pre", timeout=15000)
                    topic_data = json.loads(page.inner_text("pre"))

                    # Filter posts within the specified date range for this specific topic
                    posts_to_save = []
                    for post in topic_data.get("post_stream", {}).get("posts", []):
                        post_created_at_str = post.get('created_at')
                        if post_created_at_str:
                            try:
                                post_created_at = parse_date(post_created_at_str)
                                if START_DATE <= post_created_at <= END_DATE:
                                    # Clean HTML content of the post
                                    if "cooked" in post:
                                        post["cleaned_content"] = BeautifulSoup(post["cooked"], "html.parser").get_text(separator=' ').strip()
                                        post["cleaned_content"] = re.sub(r'\s+', ' ', post["cleaned_content"]).strip()
                                    else:
                                        post["cleaned_content"] = "" # Ensure cleaned_content exists
                                    posts_to_save.append(post)
                            except ValueError:
                                logger.warning(f"Invalid date format for post {post.get('id')} in topic {topic_id}: {post_created_at_str}")

                    # Update topic_data to only include filtered posts
                    topic_data['post_stream']['posts'] = posts_to_save

                    if posts_to_save:
                        # Save topic-style JSON file
                        filename = f"{topic_slug}_{topic_id}.json"
                        filepath = os.path.join(DOWNLOAD_DIR, filename)
                        with open(filepath, "w", encoding="utf-8") as f:
                            json.dump(topic_data, f, indent=2)
                        saved_posts_count += len(posts_to_save)
                        logger.info(f"Saved {len(posts_to_save)} posts from topic {topic_id} to {filepath}")
                    else:
                        logger.debug(f"No relevant posts found in topic {topic_id} within the date range after filtering. Skipping save.")

                else:
                    logger.debug(f"Topic {topic_id} ({created_at_str}) is outside the date range. Skipping.")
                
            except (TimeoutError, json.JSONDecodeError, Error) as e:
                logger.error(f"❌ Error fetching or processing topic {topic_id} ({topic_summary.get('title')}): {e}")
                logger.debug(traceback.format_exc()) # Print full traceback for Playwright errors
            except Exception as e:
                logger.error(f"An unexpected error occurred while processing topic {topic_id}: {e}")
                logger.error(traceback.format_exc()) # Print full traceback for unexpected errors
            
            time.sleep(0.5) # Be kind to the server, add a small delay between topic detail fetches

    except Exception as e:
        logger.critical(f"Critical error during scraping process (top-level catch): {e}")
        logger.error(traceback.format_exc())
    finally:
        if browser:
            browser.close()
            logger.info("Browser closed by scrape_posts.")
    
    logger.info(f"Discourse scraping complete. Total posts saved: {saved_posts_count}.")
    if saved_posts_count == 0:
        logger.warning("No Discourse posts were saved. Please ensure login was successful, dates are correct, and content is available.")


def main_orchestrator():
    """
    Main function to orchestrate the Playwright browser context and scraping.
    """
    with sync_playwright() as p:
        context = None
        current_browser = None 
        try:
            if os.path.exists(AUTH_STATE_FILE) and os.path.getsize(AUTH_STATE_FILE) > 0: # Check if file exists and is not empty
                logger.info(f"Found existing auth.json. Attempting to use saved session: {os.path.getsize(AUTH_STATE_FILE)} bytes.")
                current_browser = p.chromium.launch(headless=True)
                context = current_browser.new_context(storage_state=AUTH_STATE_FILE)
                if not is_authenticated(context):
                    logger.warning("Existing session invalid or expired. Re-authenticating.")
                    if context: context.close()
                    if current_browser: current_browser.close()
                    login_and_save_auth(p) # Re-authenticate
                    # After manual login, re-load the context with the new state
                    current_browser = p.chromium.launch(headless=True)
                    context = current_browser.new_context(storage_state=AUTH_STATE_FILE)
                    if not is_authenticated(context): # Final check after re-auth
                        logger.critical("Failed to authenticate even after re-login. Exiting.")
                        if current_browser: current_browser.close()
                        return
                else:
                    logger.info("✅ Using existing authenticated session.")
            else:
                logger.info("No valid auth.json found. Performing initial manual login.")
                login_and_save_auth(p) # No auth file, perform initial login
                # After manual login, create context with the new state
                current_browser = p.chromium.launch(headless=True)
                context = current_browser.new_context(storage_state=AUTH_STATE_FILE)
                if not is_authenticated(context): # Final check after login
                    logger.critical("Failed to authenticate immediately after login. Exiting.")
                    if current_browser: current_browser.close()
                    return

            # If we reached here, we have a valid, authenticated context
            scrape_posts(p) # Call scrape_posts with the playwright instance

        except Error as e: # Catch Playwright-specific errors during launch/context creation
            logger.critical(f"Playwright error during browser launch or context creation: {e}")
            logger.error(traceback.format_exc())
        except Exception as e:
            logger.critical(f"Unhandled error in main_orchestrator: {e}")
            logger.error(traceback.format_exc())
        finally:
            if current_browser:
                current_browser.close() # Ensure browser is closed if it was launched
                logger.info("Browser closed by orchestrator.")


if __name__ == "__main__":
    main_orchestrator()

