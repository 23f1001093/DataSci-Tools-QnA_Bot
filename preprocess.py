import os
import json
import sqlite3
import re
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import aiohttp
import asyncio
import argparse
import markdown
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
import openai # This import might not be strictly needed if only using AIPipe proxy, but harmless.
import traceback # Import traceback for detailed error logging

# Load environment variables from .env
load_dotenv(".env")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Read AIPipe credentials
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
AIPIPE_BASE_URL = os.getenv("AIPIPE_BASE_URL", "https://aipipe.org/openai/v1")

if not AIPIPE_TOKEN:
    logger.error("❌ AIPIPE_TOKEN not found in .env file. Please set it to proceed with embeddings.")
    # Do not exit here, allow script to run partially for data loading if needed for debugging
    # but embedding creation will fail.

# Function to call AIPipe embedding API (Synchronous version for initial testing)
# This synchronous version is not used in create_embeddings, but is kept for quick local tests
def get_embedding_sync(text):
    url = f"{AIPIPE_BASE_URL}/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPIPE_TOKEN}"
    }
    data = {
        "model": "text-embedding-3-small",
        "input": text
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        embedding = response.json()["data"][0]["embedding"]
        logger.debug(f"✅ Got embedding for input: {text[:30]}... Vector (first 5): {embedding[:5]}")
        return embedding
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Synchronous embedding failed: {e}. Response: {response.text if 'response' in locals() else 'N/A'}")
        return None

# Paths
# CORRECTED: Point DISCOURSE_DIR to the 'scraping' subdirectory
DISCOURSE_DIR = "scraping/downloaded_threads" 
MARKDOWN_DIR = "markdown_files" # Assuming markdown_files is in the root
DB_PATH = "knowledge_base.db"

# Ensure markdown_files directory exists (scrape_discourse.py handles downloaded_threads)
os.makedirs(MARKDOWN_DIR, exist_ok=True)
logger.info(f"Ensured '{MARKDOWN_DIR}' directory exists.")

# Chunking parameters (can be overridden by command-line arguments)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# SQLite connection
def create_connection():
    """
    Establishes a connection to the SQLite database.
    Sets row_factory to sqlite3.Row for dictionary-like access to rows.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row # Enable dictionary-like access to rows
        logger.info(f"Connected to SQLite database at {DB_PATH}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database: {e}")
        return None

# Create required database tables
def create_tables(conn):
    """
    Creates the necessary tables for Discourse and Markdown chunks if they don't exist.
    """
    try:
        cursor = conn.cursor()

        # Table for Discourse chunks
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS discourse_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id INTEGER,
                topic_id INTEGER,
                topic_title TEXT,
                post_number INTEGER,
                author TEXT,
                created_at TEXT,
                likes INTEGER,
                chunk_index TEXT,
                content TEXT,
                url TEXT,
                embedding BLOB
            )
        ''')

        # Table for Markdown chunks
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS markdown_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_title TEXT,
                original_url TEXT,
                downloaded_at TEXT,
                chunk_index TEXT,
                content TEXT,
                embedding BLOB
            )
        ''')

        conn.commit()
        logger.info("✅ Database tables created successfully.")
    except sqlite3.Error as e:
        logger.error(f"❌ Error creating tables: {e}")
        logger.error(traceback.format_exc())

# Split text into overlapping chunks with improved chunking
def create_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Splits a given text into overlapping chunks based on paragraph and sentence boundaries.
    Args:
        text (str): The input text to chunk.
        chunk_size (int): The desired maximum size of each chunk.
        chunk_overlap (int): The number of characters to overlap between consecutive chunks.
    Returns:
        list: A list of text chunks.
    """
    if not text:
        return []
    
    chunks = []
    
    # Clean up whitespace and newlines, preserving meaningful paragraph breaks
    text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with single
    text = re.sub(r'\s+', ' ', text)   # Replace multiple spaces with single
    text = text.strip()
    
    # If text is very short, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]
    
    # Split text by paragraphs for more meaningful chunks
    paragraphs = text.split('\n')
    current_chunk = ""
    
    for i, para in enumerate(paragraphs):
        # If this paragraph alone exceeds chunk size, we need to split it further
        if len(para) > chunk_size:
            # If we have content in the current chunk, store it first
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split the paragraph into sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sentence_chunk = ""
            
            for sentence in sentences:
                # If single sentence exceeds chunk size, split by chunks with overlap
                if len(sentence) > chunk_size:
                    # If we have content in the sentence chunk, store it first
                    if sentence_chunk:
                        chunks.append(sentence_chunk.strip())
                        sentence_chunk = ""
                    
                    # Process the long sentence in chunks
                    for j in range(0, len(sentence), chunk_size - chunk_overlap):
                        sentence_part = sentence[j:j + chunk_size]
                        if sentence_part:
                            chunks.append(sentence_part.strip())
                
                # If adding this sentence would exceed chunk size, save current and start new
                elif len(sentence_chunk) + len(sentence) > chunk_size and sentence_chunk:
                    chunks.append(sentence_chunk.strip())
                    sentence_chunk = sentence
                else:
                    # Add to current sentence chunk
                    if sentence_chunk:
                        sentence_chunk += " " + sentence
                    else:
                        sentence_chunk = sentence
            
            # Add any remaining sentence chunk
            if sentence_chunk:
                chunks.append(sentence_chunk.strip())
            
        # Normal paragraph handling - if adding would exceed chunk size
        elif len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with this paragraph
            current_chunk = para
        else:
            # Add to current chunk
            if current_chunk:
                current_chunk += " " + para
            else:
                current_chunk = para
    
    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Apply overlap between chunks to ensure continuity if chunks were made
    overlapped_chunks = []
    if chunks:
        # Start with the first chunk
        overlapped_chunks.append(chunks[0])
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk_original = chunks[i] # Store original current chunk
            
            # Determine overlap section from the end of the previous chunk
            overlap_section = prev_chunk[max(0, len(prev_chunk) - chunk_overlap):]
            
            # Prepend overlap to the current chunk if it's not already there
            if not current_chunk_original.startswith(overlap_section):
                # Try to find a natural break point (like end of sentence) within overlap_section
                # to make the overlap more meaningful, otherwise just use the raw characters.
                last_sentence_end_in_overlap = overlap_section.rfind('. ')
                if last_sentence_end_in_overlap != -1:
                    actual_overlap_to_add = overlap_section[last_sentence_end_in_overlap + 2:]
                    if actual_overlap_to_add and not current_chunk_original.startswith(actual_overlap_to_add):
                        current_chunk = actual_overlap_to_add + " " + current_chunk_original
                    else:
                        current_chunk = current_chunk_original # No meaningful sentence part to add, or already there
                else:
                    current_chunk = overlap_section + " " + current_chunk_original # Just add the raw overlap
            else:
                current_chunk = current_chunk_original # Overlap already present or no good overlap section
            
            overlapped_chunks.append(current_chunk.strip())
        
        return overlapped_chunks
    
    # If no chunks were created but text exists (e.g., after extensive cleaning), return it as a single chunk
    if text:
        return [text]
    
    return []

# Clean HTML content from Discourse posts
def clean_html(html_content):
    """
    Cleans HTML content by removing scripts, styles, and extra whitespace.
    Args:
        html_content (str): The HTML string.
    Returns:
        str: Cleaned plain text content.
    """
    if not html_content:
        return ""
    
    # Use BeautifulSoup to parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    
    # Convert to text and clean up whitespace
    text = soup.get_text(separator=' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Parse Discourse JSON files
def process_discourse_files(conn):
    """
    Processes downloaded Discourse JSON files, extracts post data,
    chunks content, and stores it in the database.
    Args:
        conn (sqlite3.Connection): The database connection object.
    """
    cursor = conn.cursor()
    
    # Check if table exists and has data to avoid re-processing
    cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
    count = cursor.fetchone()[0]
    if count > 0:
        logger.info(f"Found {count} existing discourse chunks in database, skipping processing Discourse files.")
        return
    
    # Check if DISCOURSE_DIR exists before listing its contents
    if not os.path.exists(DISCOURSE_DIR):
        logger.warning(f"Discourse directory '{DISCOURSE_DIR}' not found. Skipping Discourse file processing.")
        return

    discourse_files = [f for f in os.listdir(DISCOURSE_DIR) if f.endswith('.json')]
    logger.info(f"Found {len(discourse_files)} Discourse JSON files to process in '{DISCOURSE_DIR}'.")
    
    total_chunks = 0
    
    for file_name in tqdm(discourse_files, desc="Processing Discourse files"):
        try:
            file_path = os.path.join(DISCOURSE_DIR, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                # Extract topic information
                topic_id = data.get('id')
                topic_title = data.get('title', '')
                topic_slug = data.get('slug', '')
                
                # Process each post in the topic
                posts = data.get('post_stream', {}).get('posts', [])
                
                for post in posts:
                    post_id = post.get('id')
                    post_number = post.get('post_number')
                    author = post.get('username', '')
                    created_at = post.get('created_at', '')
                    
                    # Use the 'cleaned_content' if available from the scraper, otherwise fall back to 'cooked'
                    clean_content = post.get('cleaned_content') 
                    if clean_content is None: # Fallback if scraper didn't save cleaned_content
                         html_content = post.get('cooked', '')
                         clean_content = clean_html(html_content)
                    
                    # Skip if content is too short (e.g., just a signature or empty post)
                    if len(clean_content) < 20:
                        continue
                    
                    # Create post URL with proper format
                    url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_slug}/{topic_id}/{post_number}"
                    
                    # Split content into chunks
                    chunks = create_chunks(clean_content)
                    
                    # Store chunks in database
                    for i, chunk in enumerate(chunks):
                        cursor.execute('''
                        INSERT INTO discourse_chunks 
                        (post_id, topic_id, topic_title, post_number, author, created_at, likes, chunk_index, content, url, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (post_id, topic_id, topic_title, post_number, author, created_at, post.get('like_count', 0), i, chunk, url, None))
                        total_chunks += 1
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error processing Discourse file {file_name}: {e}")
            logger.error(traceback.format_exc()) # Log full traceback
    
    logger.info(f"Finished processing Discourse files. Created {total_chunks} chunks.")

# Parse markdown files
def process_markdown_files(conn):
    """
    Processes downloaded Markdown files, extracts content and metadata,
    chunks content, and stores it in the database.
    Args:
        conn (sqlite3.Connection): The database connection object.
    """
    cursor = conn.cursor()
    
    # Check if table exists and has data to avoid re-processing
    cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
    count = cursor.fetchone()[0]
    if count > 0:
        logger.info(f"Found {count} existing markdown chunks in database, skipping processing Markdown files.")
        return
    
    # Check if MARKDOWN_DIR exists before listing its contents
    if not os.path.exists(MARKDOWN_DIR):
        logger.warning(f"Markdown directory '{MARKDOWN_DIR}' not found. Skipping Markdown file processing.")
        return

    markdown_files = [f for f in os.listdir(MARKDOWN_DIR) if f.endswith('.md')]
    logger.info(f"Found {len(markdown_files)} Markdown files to process in '{MARKDOWN_DIR}'.")
    
    total_chunks = 0
    
    for file_name in tqdm(markdown_files, desc="Processing Markdown files"):
        try:
            file_path = os.path.join(MARKDOWN_DIR, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Extract metadata from frontmatter
                title = ""
                original_url = ""
                downloaded_at = ""
                
                # Extract metadata from frontmatter if present (YAML-like block at the top)
                frontmatter_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
                if frontmatter_match:
                    frontmatter = frontmatter_match.group(1)
                    
                    # Extract title
                    title_match = re.search(r'title:\s*["\']?(.*?)["\']?$', frontmatter, re.MULTILINE)
                    if title_match:
                        title = title_match.group(1).strip()
                    
                    # Extract original URL
                    url_match = re.search(r'original_url:\s*["\']?(.*?)["\']?$', frontmatter, re.MULTILINE)
                    if url_match:
                        original_url = url_match.group(1).strip()
                    
                    # Extract downloaded at timestamp
                    date_match = re.search(r'downloaded_at:\s*["\']?(.*?)["\']?$', frontmatter, re.MULTILINE)
                    if date_match:
                        downloaded_at = date_match.group(1).strip()
                    
                    # Remove frontmatter from content
                    content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
                
                # Split content into chunks
                chunks = create_chunks(content)
                
                # Store chunks in database
                for i, chunk in enumerate(chunks):
                    cursor.execute('''
                    INSERT INTO markdown_chunks 
                    (doc_title, original_url, downloaded_at, chunk_index, content, embedding)
                    VALUES (?, ?, ?, ?, ?, NULL)
                    ''', (title, original_url, downloaded_at, i, chunk))
                    total_chunks += 1
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error processing Markdown file {file_name}: {e}")
            logger.error(traceback.format_exc()) # Log full traceback
    
    logger.info(f"Finished processing Markdown files. Created {total_chunks} chunks.")

# Function to create embeddings using aipipe proxy with improved error handling and retries
async def create_embeddings(api_key):
    """
    Asynchronously creates embeddings for un-embedded chunks in the database
    using the AIPipe proxy, handling long texts, rate limits, and retries.
    Args:
        api_key (str): The API key for AIPipe.
    """
    if not api_key:
        logger.error("API key is missing. Cannot create embeddings.")
        return
        
    conn = create_connection()
    if conn is None:
        logger.error("Failed to connect to database for embedding creation.")
        return

    cursor = conn.cursor()
    
    # Get discourse chunks without embeddings
    cursor.execute("SELECT id, content FROM discourse_chunks WHERE embedding IS NULL")
    discourse_chunks_to_embed = cursor.fetchall()
    logger.info(f"Found {len(discourse_chunks_to_embed)} discourse chunks to embed.")
    
    # Get markdown chunks without embeddings
    cursor.execute("SELECT id, content FROM markdown_chunks WHERE embedding IS NULL")
    markdown_chunks_to_embed = cursor.fetchall()
    logger.info(f"Found {len(markdown_chunks_to_embed)} markdown chunks to embed.")
    
    # Function to handle long texts by breaking them into multiple embeddings
    async def handle_long_text(session, text, record_id, is_discourse=True, max_retries=3):
        # Model limit is 8191 tokens for text-embedding-3-small
        max_chars = 8000  # Conservative limit to stay under token limit, considering multi-byte chars
        
        # If text is within limit, embed it directly
        if len(text) <= max_chars:
            return await embed_text(session, text, record_id, is_discourse, max_retries)
        
        # For long texts, we need to split and create multiple embeddings
        logger.info(f"Text exceeds embedding limit for record {record_id}: {len(text)} chars. Creating multiple embeddings.")
        
        # Get original chunk data to duplicate for multi-part embeddings
        original_chunk_data = None
        if is_discourse:
            cursor.execute("""
            SELECT post_id, topic_id, topic_title, post_number, author, created_at, 
                   likes, chunk_index, content, url FROM discourse_chunks 
            WHERE id = ?
            """, (record_id,))
            original_chunk_data = cursor.fetchone()
        else:
            cursor.execute("""
            SELECT doc_title, original_url, downloaded_at, chunk_index FROM markdown_chunks 
            WHERE id = ?
            """, (record_id,))
            original_chunk_data = cursor.fetchone()

        if not original_chunk_data:
            logger.error(f"Original chunk data not found for record ID {record_id}. Cannot process long text.")
            return False

        # Create overlapping subchunks for the long text
        overlap_for_long_text_split = 200 # Small overlap between subchunks
        subchunks = []
        for i in range(0, len(text), max_chars - overlap_for_long_text_split):
            end = min(i + max_chars, len(text))
            subchunk = text[i:end]
            if subchunk:
                subchunks.append(subchunk)
        
        logger.info(f"Split into {len(subchunks)} subchunks for embedding record {record_id}")
        
        all_succeeded = True
        # Delete the original large chunk record before inserting new ones
        cursor.execute(f"DELETE FROM {'discourse_chunks' if is_discourse else 'markdown_chunks'} WHERE id = ?", (record_id,))
        conn.commit()
        logger.debug(f"Deleted original large chunk {record_id} before splitting and embedding.")


        for i, subchunk in enumerate(subchunks):
            current_part_id = f"part_{i+1}_of_{len(subchunks)}"
            logger.info(f"Embedding subchunk {i+1}/{len(subchunks)} for record {record_id} ({current_part_id})")
            
            # Insert a new record for each subchunk with its own content and embedding
            if is_discourse:
                try:
                    cursor.execute('''
                    INSERT INTO discourse_chunks 
                    (post_id, topic_id, topic_title, post_number, author, created_at, likes, chunk_index, content, url, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        original_chunk_data["post_id"], 
                        original_chunk_data["topic_id"], 
                        original_chunk_data["topic_title"], 
                        original_chunk_data["post_number"],
                        original_chunk_data["author"], 
                        original_chunk_data["created_at"], 
                        original_chunk_data["likes"], 
                        f"{original_chunk_data['chunk_index']}_{current_part_id}",  # Append part_id to chunk_index
                        subchunk, # Use the subchunk content
                        original_chunk_data["url"], 
                        None # Will be updated with actual embedding
                    ))
                    new_record_id = cursor.lastrowid # Get the ID of the newly inserted record
                    conn.commit()
                    success = await embed_text(
                        session, 
                        subchunk, 
                        new_record_id, # Use the new record ID
                        is_discourse, 
                        max_retries
                    )
                except Exception as e:
                    logger.error(f"Error inserting or embedding discourse subchunk for record {record_id}, part {current_part_id}: {e}")
                    logger.error(traceback.format_exc())
                    success = False
            else:
                try:
                    cursor.execute('''
                    INSERT INTO markdown_chunks 
                    (doc_title, original_url, downloaded_at, chunk_index, content, embedding)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        original_chunk_data["doc_title"],
                        original_chunk_data["original_url"],
                        original_chunk_data["downloaded_at"],
                        f"{original_chunk_data['chunk_index']}_{current_part_id}", # Append part_id to chunk_index
                        subchunk, # Use the subchunk content
                        None # Will be updated with actual embedding
                    ))
                    new_record_id = cursor.lastrowid # Get the ID of the newly inserted record
                    conn.commit()
                    success = await embed_text(
                        session, 
                        subchunk, 
                        new_record_id, # Use the new record ID
                        is_discourse, 
                        max_retries
                    )
                except Exception as e:
                    logger.error(f"Error inserting or embedding markdown subchunk for record {record_id}, part {current_part_id}: {e}")
                    logger.error(traceback.format_exc())
                    success = False

            if not success:
                all_succeeded = False
                logger.error(f"Failed to embed subchunk {i+1}/{len(subchunks)} for original record {record_id}")
        
        return all_succeeded
    
    # Function to embed a single text with retry mechanism
    async def embed_text(session, text, record_id, is_discourse=True, max_retries=3):
        """
        Attempts to embed a single text chunk with retries on rate limits.
        Updates the database with the embedding.
        Args:
            session (aiohttp.ClientSession): The aiohttp client session.
            text (str): The text to embed.
            record_id (int): The ID of the database record to update.
            is_discourse (bool): True if it's a discourse chunk, False for markdown.
            max_retries (int): Maximum number of retries for API calls.
        Returns:
            bool: True if embedding was successful and saved, False otherwise.
        """
        retries = 0
        while retries < max_retries:
            try:
                # Call the embedding API through aipipe proxy
                url = f"{AIPIPE_BASE_URL}/embeddings" # Use AIPIPE_BASE_URL
                headers = {
                    # CORRECTED: Added "Bearer " prefix to Authorization header
                    "Authorization": f"Bearer {api_key}", 
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "text-embedding-3-small",
                    "input": text
                }
                
                async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        embedding = result["data"][0]["embedding"]
                        
                        # Convert embedding to binary blob for SQLite storage
                        embedding_blob = json.dumps(embedding).encode()
                        
                        # Update the database record with the new embedding
                        if is_discourse:
                            cursor.execute(
                                "UPDATE discourse_chunks SET embedding = ? WHERE id = ?",
                                (embedding_blob, record_id)
                            )
                        else:
                            cursor.execute(
                                "UPDATE markdown_chunks SET embedding = ? WHERE id = ?",
                                (embedding_blob, record_id)
                            )
                        
                        conn.commit()
                        logger.info(f"✅ Embedded and saved for record ID {record_id}.")
                        return True
                    elif response.status == 429:  # Rate limit error
                        error_text = await response.text()
                        logger.warning(f"Rate limit reached for record ID {record_id}, retrying after delay (retry {retries+1}): {error_text}")
                        await asyncio.sleep(5 * (retries + 1))  # Exponential backoff
                        retries += 1
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Error embedding text for record ID {record_id} (status {response.status}): {error_text}")
                        return False
            except aiohttp.ClientError as e: # Catch network/client specific errors
                logger.error(f"Aiohttp client error embedding text for record ID {record_id}: {e}")
                retries += 1
                await asyncio.sleep(3 * retries)  # Wait before retry
            except Exception as e: # Catch any other unexpected errors
                logger.error(f"Unexpected exception embedding text for record ID {record_id}: {e}")
                retries += 1
                await asyncio.sleep(3 * retries) # Wait before retry
        
        logger.error(f"Failed to embed text for record ID {record_id} after {max_retries} retries.")
        return False
    
    # Process in smaller batches to avoid rate limits and manage memory
    batch_size = 10
    async with aiohttp.ClientSession() as session:
        # Process discourse chunks
        for i in tqdm(range(0, len(discourse_chunks_to_embed), batch_size), desc="Embedding Discourse chunks"):
            batch = discourse_chunks_to_embed[i:i+batch_size]
            tasks = [handle_long_text(session, text, record_id, True) for record_id, text in batch]
            await asyncio.gather(*tasks) # Await all tasks in the batch
            
            # Sleep to avoid rate limits between batches
            if i + batch_size < len(discourse_chunks_to_embed):
                await asyncio.sleep(2) # Short delay between batches

        # Process markdown chunks
        for i in tqdm(range(0, len(markdown_chunks_to_embed), batch_size), desc="Embedding Markdown chunks"):
            batch = markdown_chunks_to_embed[i:i+batch_size]
            tasks = [handle_long_text(session, text, record_id, False) for record_id, text in batch]
            await asyncio.gather(*tasks) # Await all tasks in the batch
            
            # Sleep to avoid rate limits between batches
            if i + batch_size < len(markdown_chunks_to_embed):
                await asyncio.sleep(2) # Short delay between batches
    
    conn.close()
    logger.info("Finished creating embeddings.")

# Main function orchestrating the preprocessing pipeline
async def main():
    """
    Main asynchronous function to run the data preprocessing pipeline.
    Handles command-line arguments, database setup, file processing, and embedding creation.
    """
    global CHUNK_SIZE, CHUNK_OVERLAP # Declare global to modify
    
    parser = argparse.ArgumentParser(description="Preprocess Discourse posts and markdown files for RAG system")
    parser.add_argument("--api-key", help="API key for aipipe proxy (if not provided, will use AIPIPE_TOKEN environment variable)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help=f"Size of text chunks (default: {CHUNK_SIZE})")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help=f"Overlap between chunks (default: {CHUNK_OVERLAP})")
    args = parser.parse_args()
    
    # Get API key from arguments or environment variable (AIPIPE_TOKEN)
    api_key = args.api_key or AIPIPE_TOKEN 
    if not api_key:
        logger.error("API key not provided. Please provide it via --api-key argument or set the AIPIPE_TOKEN environment variable.")
        return
    
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap
    
    logger.info(f"Using chunk size: {CHUNK_SIZE}, chunk overlap: {CHUNK_OVERLAP}")
    
    # Create database connection
    conn = create_connection()
    if conn is None:
        return # Exit if connection fails
    
    # Create tables
    create_tables(conn)
    
    # Process files
    process_discourse_files(conn)
    process_markdown_files(conn)
    
    # Create embeddings
    await create_embeddings(api_key)
    
    # Close connection
    conn.close()
    logger.info("Preprocessing complete.")

if __name__ == "__main__":
    # Run the main asynchronous function
    asyncio.run(main())

