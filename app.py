import os
import json
import sqlite3
import numpy as np
import re
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import aiohttp
import asyncio
import logging
import base64
from fastapi.responses import JSONResponse
import uvicorn
import traceback
from dotenv import load_dotenv
from urllib.parse import urlparse # Import urlparse for URL manipulation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DB_PATH = "knowledge_base.db"
SIMILARITY_THRESHOLD = 0.20  # LOWERED THRESHOLD FOR BETTER RECALL
MAX_RESULTS = 10  # Increased to get more context (total number of chunks from all sources)
load_dotenv()
MAX_CONTEXT_CHUNKS = 4  # Increased number of chunks per unique source document/post
API_KEY = os.getenv("AIPIPE_TOKEN") # IMPORTANT: Using AIPIPE_TOKEN for consistency with preprocess.py

# Models for FastAPI request/response
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # Base64 encoded image

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]

# Initialize FastAPI app
app = FastAPI(title="RAG Query API", description="API for querying the RAG knowledge base")

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Verify API key is set
if not API_KEY:
    logger.error("AIPIPE_TOKEN environment variable is not set. The application will not function correctly.")

# Create a connection to the SQLite database
def get_db_connection():
    """
    Establishes and returns a SQLite database connection.
    Sets row_factory to sqlite3.Row for dictionary-like column access.
    Raises HTTPException if connection fails.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # This enables column access by name (e.g., row['column_name'])
        return conn
    except sqlite3.Error as e:
        error_msg = f"Database connection error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc()) # Log full traceback for debugging
        raise HTTPException(status_code=500, detail=error_msg)

# Make sure database tables exist or create them if the database file is new
# This block runs once when the application starts
if not os.path.exists(DB_PATH):
    logger.info(f"Database file {DB_PATH} not found. Creating tables.")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Create discourse_chunks table
    c.execute('''
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
    
    # Create markdown_chunks table
    c.execute('''
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
    conn.close()
    logger.info("Database tables created successfully for new DB file.")

# Vector similarity calculation (Cosine Similarity)
def cosine_similarity(vec1, vec2):
    """
    Calculates the cosine similarity between two vectors.
    Handles zero vectors to prevent division by zero.
    """
    try:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Handle zero vectors which can occur if embedding failed or is malformed
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
            
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        # Avoid division by zero if norms are effectively zero
        if norm_vec1 < 1e-9 or norm_vec2 < 1e-9: # Using a small epsilon for floating point comparison
            return 0.0
            
        return dot_product / (norm_vec1 * norm_vec2)
    except Exception as e:
        logger.error(f"Error in cosine_similarity calculation: {e}")
        logger.error(traceback.format_exc())
        return 0.0  # Return 0 similarity on error rather than crashing the request

# Function to get embedding from aipipe proxy with retry mechanism
async def get_embedding(text, max_retries=3):
    """
    Asynchronously obtains an embedding for the given text using the AIPipe proxy.
    Includes retry logic for rate limits and other temporary errors.
    """
    if not API_KEY:
        error_msg = "AIPIPE_TOKEN environment variable not set. Cannot get embedding."
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    retries = 0
    while retries < max_retries:
        try:
            logger.info(f"Attempting to get embedding for text (length: {len(text)}) (Retry: {retries+1})")
            url = "https://aipipe.org/openai/v1/embeddings"
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "text-embedding-3-small",
                "input": text
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Successfully received embedding.")
                        return result["data"][0]["embedding"]
                    elif response.status == 429:  # Rate limit error
                        error_text = await response.text()
                        logger.warning(f"Rate limit reached for embedding API. Retrying after delay (retry {retries+1}): {error_text}")
                        await asyncio.sleep(5 * (retries + 1))  # Exponential backoff
                        retries += 1
                    else:
                        error_text = await response.text()
                        error_msg = f"Error getting embedding from AIPipe (status {response.status}): {error_text}"
                        logger.error(error_msg)
                        raise HTTPException(status_code=response.status, detail=error_msg)
        except aiohttp.ClientError as e: # Catch network/client specific errors (e.g., connection issues, timeouts)
            error_msg = f"Aiohttp client error during embedding request (attempt {retries+1}/{max_retries}): {e}"
            logger.error(error_msg)
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(3 * retries)  # Wait before retry
        except Exception as e: # Catch any other unexpected errors
            error_msg = f"Unexpected exception getting embedding (attempt {retries+1}/{max_retries}): {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(3 * retries)  # Wait before retry

# Function to find similar content in the database
async def find_similar_content(query_embedding, conn):
    """
    Finds the most similar content chunks in the database based on cosine similarity
    with the given query embedding.
    Groups results by source document/post and selects top chunks per source.
    """
    try:
        logger.info("Starting search for similar content in database.")
        cursor = conn.cursor()
        results = []
        
        # Search discourse chunks
        logger.info("Querying discourse chunks for embeddings.")
        cursor.execute("""
        SELECT id, post_id, topic_id, topic_title, post_number, author, created_at, 
               likes, chunk_index, content, url, embedding 
        FROM discourse_chunks 
        WHERE embedding IS NOT NULL
        """)
        
        discourse_chunks = cursor.fetchall()
        logger.info(f"Processing {len(discourse_chunks)} discourse chunks.")
        
        for chunk in discourse_chunks:
            try:
                # Ensure embedding is parsed from JSON blob
                embedding = json.loads(chunk["embedding"])
                similarity = cosine_similarity(query_embedding, embedding)
                
                if similarity >= SIMILARITY_THRESHOLD:
                    # Robustly construct Discourse URL using available components
                    topic_id = chunk["topic_id"]
                    post_number = chunk["post_number"]
                    topic_title = chunk["topic_title"]
                    
                    # Generate slug for URL if topic_title exists
                    topic_slug = ""
                    if topic_title:
                        topic_slug = re.sub(r'[^a-zA-Z0-9-]', '', topic_title.lower().replace(' ', '-'))
                        
                    # Prioritize slug if available, otherwise use just topic_id
                    if topic_slug:
                        full_url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_slug}/{topic_id}/{post_number}"
                    else:
                        full_url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_id}/{post_number}"

                    results.append({
                        "source": "discourse",
                        "id": chunk["id"],
                        "post_id": chunk["post_id"],
                        "topic_id": chunk["topic_id"],
                        "title": chunk["topic_title"],
                        "url": full_url, # Use the robustly constructed URL
                        "content": chunk["content"],
                        "author": chunk["author"],
                        "created_at": chunk["created_at"],
                        "chunk_index": chunk["chunk_index"],
                        "similarity": float(similarity)
                    })
                
            except json.JSONDecodeError:
                logger.error(f"Failed to decode embedding JSON for discourse chunk ID {chunk['id']}. Skipping.")
            except Exception as e:
                logger.error(f"Error processing discourse chunk {chunk['id']}: {e}")
                logger.error(traceback.format_exc())
        
        # Search markdown chunks
        logger.info("Querying markdown chunks for embeddings.")
        cursor.execute("""
        SELECT id, doc_title, original_url, downloaded_at, chunk_index, content, embedding 
        FROM markdown_chunks 
        WHERE embedding IS NOT NULL
        """)
        
        markdown_chunks = cursor.fetchall()
        logger.info(f"Processing {len(markdown_chunks)} markdown chunks.")
        
        for chunk in markdown_chunks:
            try:
                embedding = json.loads(chunk["embedding"])
                similarity = cosine_similarity(query_embedding, embedding)
                
                if similarity >= SIMILARITY_THRESHOLD:
                    # Ensure URL is properly formatted, default if missing or incomplete
                    url = chunk["original_url"]
                    
                    # If URL exists but doesn't start with http, prepend https://
                    if url and not url.startswith("http"):
                        url = f"https://{url}"
                    elif not url: # If original_url is completely missing, then use the fallback
                        # Construct a plausible URL if original_url is missing or invalid
                        clean_title = re.sub(r'[^a-zA-Z0-9-]', '', chunk['doc_title'].lower().replace(' ', '-'))
                        url = f"https://docs.onlinedegree.iitm.ac.in/{clean_title}.md" # Assuming .md files or similar structure
                    
                    results.append({
                        "source": "markdown",
                        "id": chunk["id"],
                        "title": chunk["doc_title"],
                        "url": url, # Use the correctly formatted URL
                        "content": chunk["content"],
                        "chunk_index": chunk["chunk_index"],
                        "similarity": float(similarity)
                    })
                
            except json.JSONDecodeError:
                logger.error(f"Failed to decode embedding JSON for markdown chunk ID {chunk['id']}. Skipping.")
            except Exception as e:
                logger.error(f"Error processing markdown chunk {chunk['id']}: {e}")
                logger.error(traceback.format_exc())
        
        # Sort all results by similarity (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        logger.info(f"Found {len(results)} relevant results above threshold.")
        
        # Group by unique source document/post and keep most relevant chunks
        grouped_results = {}
        for result in results:
            # Create a unique key for the document/post
            if result["source"] == "discourse":
                # A Discourse post is unique by topic_id and post_id
                key = f"discourse_{result.get('topic_id')}_{result.get('post_id')}"
            else: # markdown
                # A Markdown document is unique by its title/original_url
                key = f"markdown_{result.get('title') or result.get('original_url')}"
            
            if key not in grouped_results:
                grouped_results[key] = []
            
            # Add result if it doesn't exceed MAX_CONTEXT_CHUNKS per source
            if len(grouped_results[key]) < MAX_CONTEXT_CHUNKS:
                grouped_results[key].append(result)
        
        # Flatten the grouped results back into a list
        final_results = []
        for key in grouped_results:
            final_results.extend(grouped_results[key])
        
        # Sort again by similarity to ensure overall relevance order
        final_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top results, limited by MAX_RESULTS (overall number of chunks)
        # This takes the top N chunks from the top documents/posts, up to MAX_RESULTS total.
        logger.info(f"Returning {len(final_results[:MAX_RESULTS])} final results after grouping and re-sorting.")
        return final_results[:MAX_RESULTS]
    except Exception as e:
        error_msg = f"Error in find_similar_content: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise

# Function to enrich content with adjacent chunks
async def enrich_with_adjacent_chunks(conn, results):
    """
    Enriches the content of relevant chunks by adding adjacent chunks for more context.
    Looks for the immediate previous and next chunk based on chunk_index within the same source.
    """
    try:
        logger.info(f"Enriching {len(results)} results with adjacent chunks.")
        cursor = conn.cursor()
        enriched_results = []
        
        for result in results:
            enriched_result = result.copy()
            additional_content_parts = []
            
            # Parse chunk_index to handle potential 'part_id' suffixes
            try:
                # Assuming chunk_index is like 'X' or 'X_part_Y_of_Z'
                base_chunk_index_str = str(result["chunk_index"]).split('_')[0]
                base_chunk_index = int(base_chunk_index_str)
            except ValueError:
                logger.warning(f"Could not parse chunk_index '{result['chunk_index']}' to integer for ID {result['id']}. Skipping adjacent chunk enrichment for this item.")
                enriched_results.append(enriched_result) # Add original result
                continue

            # Check source type to correctly access fields and query tables
            if enriched_result["source"] == "discourse":
                post_id = enriched_result["post_id"]
                
                # Get previous chunk
                if base_chunk_index > 0:
                    cursor.execute("""
                    SELECT content FROM discourse_chunks 
                    WHERE post_id = ? AND chunk_index LIKE ?
                    ORDER BY chunk_index DESC LIMIT 1
                    """, (post_id, f"{base_chunk_index - 1}%")) # Use LIKE to match 'N' or 'N_part_...'
                    prev_chunk = cursor.fetchone()
                    if prev_chunk:
                        additional_content_parts.append(prev_chunk["content"])
                
                # Get next chunk
                cursor.execute("""
                SELECT content FROM discourse_chunks 
                WHERE post_id = ? AND chunk_index LIKE ?
                ORDER BY chunk_index ASC LIMIT 1
                """, (post_id, f"{base_chunk_index + 1}%")) # Use LIKE to match 'N+1' or 'N+1_part_...'
                next_chunk = cursor.fetchone()
                if next_chunk:
                    additional_content_parts.append(next_chunk["content"])
                
            elif enriched_result["source"] == "markdown":
                title = enriched_result["title"]
                # Use .get() with a default to avoid KeyError if 'original_url' is sometimes missing in markdown results
                original_url = enriched_result.get("original_url") 
                
                # Get previous chunk
                if base_chunk_index > 0:
                    # Added original_url to WHERE clause for more specific markdown chunk retrieval
                    cursor.execute("""
                    SELECT content FROM markdown_chunks 
                    WHERE doc_title = ? AND original_url = ? AND chunk_index LIKE ?
                    ORDER BY chunk_index DESC LIMIT 1
                    """, (title, original_url, f"{base_chunk_index - 1}%"))
                    prev_chunk = cursor.fetchone()
                    if prev_chunk:
                        additional_content_parts.append(prev_chunk["content"])
                
                # Get next chunk
                cursor.execute("""
                    SELECT content FROM markdown_chunks 
                    WHERE doc_title = ? AND original_url = ? AND chunk_index LIKE ?
                    ORDER BY chunk_index ASC LIMIT 1
                    """, (title, original_url, f"{base_chunk_index + 1}%"))
                next_chunk = cursor.fetchone()
                if next_chunk:
                    additional_content_parts.append(next_chunk["content"])
            
            # Combine the original content with adjacent content
            if additional_content_parts:
                # Add a separator to distinguish original content from adjacent
                enriched_result["content"] = f"{enriched_result['content']}\n\n[Additional Context]:\n{' '.join(additional_content_parts)}"
            
            enriched_results.append(enriched_result)
        
        logger.info(f"Successfully enriched {len(enriched_results)} results with adjacent chunks.")
        return enriched_results
    except Exception as e:
        error_msg = f"Error in enrich_with_adjacent_chunks: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise

# Function to generate an answer using LLM with improved prompt
async def generate_answer(question, relevant_results, max_retries=2):
    """
    Generates an answer to the question using an LLM, leveraging the provided context.
    Instructs the LLM to only use the context and to provide sources in a specific format.
    """
    if not API_KEY:
        error_msg = "AIPIPE_TOKEN environment variable not set. Cannot generate answer."
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    retries = 0
    while retries < max_retries:    
        try:
            logger.info(f"Generating answer for question: '{question[:50]}...' with {len(relevant_results)} context chunks.")
            context = ""
            for i, result in enumerate(relevant_results):
                source_type = "Discourse post" if result["source"] == "discourse" else "Documentation"
                # Limit content to ensure prompt fits token limits, while including enough context.
                # Project requirement for sources indicates brief quote/description, so 100-200 chars is good.
                context_snippet = result['content'][:2000] # Use a larger snippet for LLM context, up to 2000 chars.
                context += f"\n\n--- Source {i+1} ({source_type}, URL: {result['url']}) ---\nContent: {context_snippet}" # Added "Content:" before snippet
            
            # Prepare improved prompt for better control over LLM output
            prompt = f"""You are a helpful teaching assistant for IIT Madras Online Degree in Data Science.
            Answer the following question based ONLY on the provided context. If the answer is not available in the context,
            state: "I don't have enough information to answer this question."
            
            Ensure your answer is comprehensive yet concise.
            
            Context:
            {context}
            
            Question: {question}
            
            Return your response in this exact format:
            1. A comprehensive yet concise answer to the question.
            2. A "Sources:" section that lists the URLs and a brief (max 100 characters) relevant text snippet from the provided context that was used to answer the question.
            
            Sources must be in this exact format:
            Sources:
            1. URL: [exact_url_1], Text: [brief quote or description, max 100 chars]
            2. URL: [exact_url_2], Text: [brief quote or description, max 100 chars]
            
            Make sure the URLs are copied exactly from the context without any changes, and the text snippets are directly from the context.
            """
            
            logger.info("Sending request to LLM API (gpt-4o-mini via AIPipe).")
            url = "https://aipipe.org/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini", # As specified in project prompt
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that provides accurate answers based only on the provided context. Always include sources in your response with exact URLs and brief text snippets."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,  # Lower temperature for more deterministic and factual outputs
                "max_tokens": 1000 # Cap max tokens for response to manage cost and latency
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Successfully received answer from LLM.")
                        return result["choices"][0]["message"]["content"]
                    elif response.status == 429:  # Rate limit error
                        error_text = await response.text()
                        logger.warning(f"Rate limit reached for LLM API. Retrying after delay (retry {retries+1}): {error_text}")
                        await asyncio.sleep(3 * (retries + 1))  # Exponential backoff
                        retries += 1
                    else:
                        error_text = await response.text()
                        logger.error(f"Error generating answer from AIPipe (status {response.status}): {error_text}")
                        raise HTTPException(status_code=response.status, detail=error_msg)
        except aiohttp.ClientError as e: # Catch network/client specific errors
            error_msg = f"Aiohttp client error during LLM request (attempt {retries+1}/{max_retries}): {e}"
            logger.error(error_msg)
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(2 * retries) # Wait before retry
        except Exception as e:
            error_msg = f"Unexpected exception generating answer (attempt {retries+1}/{max_retries}): {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(2 * retries) # Wait before retry

# Function to process multimodal content (text + image)
async def process_multimodal_query(question, image_base64):
    """
    Processes a query that may include a base64 encoded image.
    If an image is present, it uses GPT-4o-mini (Vision) to describe the image in relation to the question,
    then combines this description with the original question to create a richer query for embedding.
    Falls back to text-only embedding if image processing fails or no image is provided.
    """
    if not API_KEY:
        error_msg = "AIPIPE_TOKEN environment variable not set. Cannot process multimodal query."
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
        
    try:
        logger.info(f"Processing multimodal query: question='{question[:50]}...', image_provided={image_base64 is not None}")
        if not image_base64:
            logger.info("No image provided in query, proceeding with text-only embedding.")
            return await get_embedding(question)
        
        logger.info("Image provided. Calling Vision API to describe image context.")
        # Call the GPT-4o-mini Vision API to process the image and question
        url = "https://aipipe.org/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Format the image for the API
        # Assuming image_base64 is raw base64 string, so prepend data URI scheme
        image_content = f"data:image/jpeg;base64,{image_base64}"
        
        payload = {
            "model": "gpt-4o-mini", # Using gpt-4o-mini for vision capabilities
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyze this image and provide a concise description of what you see that is relevant to the following question: {question}. Focus only on aspects related to the question. If the image is not relevant to the question, say 'The image does not appear relevant.'"},
                        {"type": "image_url", "image_url": {"url": image_content, "detail": "low"}} # Use "low" detail for faster processing if high detail isn't critical
                    ]
                }
            ],
            "max_tokens": 300 # Limit description length
        }
        
        async with aiohttp.ClientSession() as session:
            # Add timeout to vision API call
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=45)) as response: # Longer timeout for vision
                if response.status == 200:
                    result = await response.json()
                    image_description = result["choices"][0]["message"]["content"]
                    logger.info(f"Received image description from Vision API: '{image_description[:50]}...'")
                    
                    # Combine the original question with the image description to form the query for embedding
                    combined_query = f"Question: {question}\nImage Context: {image_description}"
                    logger.info(f"Generated combined query for embedding: '{combined_query[:100]}...'")
                    return await get_embedding(combined_query)
                else:
                    error_text = await response.text()
                    logger.error(f"Error processing image with Vision API (status {response.status}): {error_text}")
                    # Fall back to text-only query if image processing fails
                    logger.info("Falling back to text-only query due to image processing error.")
                    return await get_embedding(question)
    except aiohttp.ClientError as e:
        logger.error(f"Aiohttp client error processing multimodal query: {e}")
        logger.info("Falling back to text-only query due to network/client error.")
        return await get_embedding(question)
    except Exception as e:
        logger.error(f"Unexpected exception processing multimodal query: {e}")
        logger.error(traceback.format_exc())
        # Fall back to text-only query due to any other exception
        logger.info("Falling back to text-only query due to unexpected exception.")
        return await get_embedding(question)

# Function to parse LLM response and extract answer and sources with improved reliability
def parse_llm_response(response_text: str, relevant_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parses the LLM's raw string response to extract the main answer and a list of structured links.
    Prioritizes matching LLM-provided text snippets and URLs back to original, full URLs from relevant_results.
    Handles various formats for the "Sources:" section and extracts URLs and text snippets.
    """
    try:
        logger.info("Attempting to parse LLM response and cross-reference sources.")
        
        # Create a more robust lookup for original URLs based on content and parts of URL
        # Key: lowercased content snippet or a part of the URL (e.g., 'discourse.onlinedegree.iitm.ac.in/t/...')
        # Value: The full, correct URL
        original_url_lookup = {}
        for result in relevant_results:
            full_url = result['url']
            # Add full URL as a lookup key
            if full_url and full_url.startswith("http"):
                original_url_lookup[full_url.lower()] = full_url
            
            # Add parts of the URL (domain, path segments) as lookup keys
            try:
                parsed_url = urlparse(full_url)
                # Add netloc (domain)
                if parsed_url.netloc:
                    original_url_lookup[parsed_url.netloc.lower()] = full_url
                # Add path components
                path_segments = [s.lower() for s in parsed_url.path.split('/') if s]
                for i in range(1, len(path_segments) + 1):
                    path_prefix = '/'.join(path_segments[:i])
                    if path_prefix:
                        original_url_lookup[path_prefix] = full_url
            except Exception as e:
                logger.warning(f"Failed to parse URL '{full_url}' for lookup creation: {e}")

            # Add normalized content as lookup key
            normalized_content = re.sub(r'\s+', ' ', result['content']).strip()
            if normalized_content:
                original_url_lookup[normalized_content.lower()] = full_url

            # Add specific common truncated forms that LLMs might output for robustness
            if "discourse.onlinedegree.iitm.ac.in/t/" in full_url:
                original_url_lookup["https://t"] = full_url # Direct map for the common truncation
                original_url_lookup["https://discourse.onlinedegree.iitm.ac.in/t"] = full_url
            if "tds.s-anand.net" in full_url:
                original_url_lookup["https://tds.s-anand.net"] = full_url
                original_url_lookup["https://tds.s-anand.net/#/?id=notes".lower()] = full_url # Specific example

        answer = response_text.strip() # Default answer is the whole response
        parsed_links = []
        unique_parsed_urls = set() # To prevent duplicate URLs in the final list
        
        # Define common source headings
        source_headings = ["Sources:", "Source:", "References:", "Reference:"]
        
        # Try to split by any of the source headings
        sources_section_found = False
        for heading in source_headings:
            if heading in response_text:
                parts = response_text.split(heading, 1)
                answer = parts[0].strip()
                sources_text = parts[1].strip()
                sources_section_found = True
                
                source_lines = sources_text.split("\n")
                
                for line in source_lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Remove common list markers (1., 2., -, *, etc.)
                    line = re.sub(r'^\s*[\d\*\-\+]\.\s*', '', line)
                    
                    extracted_url_from_llm_line = ""
                    extracted_text_snippet_from_llm_line = ""

                    # Regex to find URL (e.g., URL: [url], URL: url, or raw url)
                    url_match = re.search(r'(?:URL|url):\s*\[?\s*(http[s]?:\/\/\S+?)\s*\]?|\[(http[s]?:\/\/\S+?)\]|(http[s]?:\/\/\S+)', line, re.IGNORECASE)
                    if url_match:
                        for g in url_match.groups():
                            if g and g.startswith("http"):
                                extracted_url_from_llm_line = g.strip()
                                break
                    
                    # Regex to find Text snippet (e.g., Text: [text], "text", Text: "text")
                    text_match = re.search(r'(?:Text|text):\s*["\']?(.*?)(?:["\']|\s*$)|\"(.*?)\"|\[(.*?)\]', line, re.IGNORECASE)
                    if text_match:
                        for g in text_match.groups():
                            if g:
                                extracted_text_snippet_from_llm_line = g.strip()
                                break
                    
                    final_resolved_url = ""
                    final_resolved_text = ""
                    
                    # Strategy: Prioritize matching snippet to original content, then matching LLM's URL to original URL
                    
                    # 1. Try to resolve based on extracted_text_snippet_from_llm_line (most reliable if present)
                    if extracted_text_snippet_from_llm_line:
                        normalized_snippet_for_lookup = re.sub(r'\s+', ' ', extracted_text_snippet_from_llm_line).strip().lower()
                        # Find the best match: longest key that contains the normalized snippet
                        best_match_key = ""
                        for lookup_key, full_url in original_url_lookup.items():
                            if normalized_snippet_for_lookup in lookup_key:
                                if len(lookup_key) > len(best_match_key): # Prefer longer (more specific) keys
                                    best_match_key = lookup_key
                                    final_resolved_url = full_url
                                    final_resolved_text = extracted_text_snippet_from_llm_line
                        if final_resolved_url:
                            logger.debug(f"Resolved by snippet: '{extracted_text_snippet_from_llm_line[:50]}...' -> {final_resolved_url}")

                    # 2. If URL not yet resolved, try to resolve based on extracted_url_from_llm_line
                    if not final_resolved_url and extracted_url_from_llm_line:
                        normalized_extracted_url_for_lookup = extracted_url_from_llm_line.lower()
                        
                        # Direct match for full URL
                        if normalized_extracted_url_for_lookup in original_url_lookup:
                            final_resolved_url = original_url_lookup[normalized_extracted_url_for_lookup]
                            final_resolved_text = extracted_text_snippet_from_llm_line if extracted_text_snippet_from_llm_line else "Source via exact URL match."
                            logger.debug(f"Resolved by direct URL match: '{extracted_url_from_llm_line}' -> {final_resolved_url}")
                        else:
                            # Try prefix or contains matching for the LLM-provided URL
                            best_match_key = ""
                            for lookup_key, full_url in original_url_lookup.items():
                                if normalized_extracted_url_for_lookup in lookup_key: # e.g., "https://t" in "https://discourse.onlinedegree.iitm.ac.in/t/..."
                                    if len(lookup_key) > len(best_match_key): # Prefer longer (more specific) keys
                                        best_match_key = lookup_key
                                        final_resolved_url = full_url
                                        final_resolved_text = extracted_text_snippet_from_llm_line if extracted_text_snippet_from_llm_line else "Source via partial URL match."
                            if final_resolved_url:
                                logger.debug(f"Resolved by partial URL match: '{extracted_url_from_llm_line}' -> {final_resolved_url}")

                    # 3. If still no URL, but LLM provided text snippet, try to find a URL from original results based on it
                    if not final_resolved_url and extracted_text_snippet_from_llm_line:
                        normalized_snippet_for_lookup = re.sub(r'\s+', ' ', extracted_text_snippet_from_llm_line).strip().lower()
                        for res in relevant_results:
                            if normalized_snippet_for_lookup in re.sub(r'\s+', ' ', res['content']).strip().lower():
                                final_resolved_url = res['url']
                                final_resolved_text = extracted_text_snippet_from_llm_line
                                logger.debug(f"Resolved by snippet (fallback): '{extracted_text_snippet_from_llm_line[:50]}...' -> {final_resolved_url}")
                                break


                    # Final addition to parsed_links
                    if final_resolved_url and final_resolved_url.startswith("http") and final_resolved_url not in unique_parsed_urls:
                        # Ensure the final text snippet adheres to the max 100 char limit
                        if len(final_resolved_text) > 100:
                            final_resolved_text = final_resolved_text[:97] + "..."
                        
                        parsed_links.append({"url": final_resolved_url, "text": final_resolved_text or "No specific text snippet provided."})
                        unique_parsed_urls.add(final_resolved_url)
                        logger.debug(f"Added link: {final_resolved_url}, Text: {final_resolved_text[:50]}...")
                    elif final_resolved_url and final_resolved_url.startswith("http"):
                        logger.debug(f"Skipping duplicate or invalid resolved URL: {final_resolved_url}")
                    else:
                        logger.debug(f"Could not resolve valid URL for LLM line: '{line}'. Extracted URL: '{extracted_url_from_llm_line}', Snippet: '{extracted_text_snippet_from_llm_line}'")

                break # Stop after finding and processing the first "Sources:" section

        # Fallback for links extraction: if LLM didn't format sources correctly OR
        # if the LLM didn't provide any structured sources, generate them from top relevant results directly.
        if not sources_section_found or not parsed_links:
            logger.warning("LLM did not provide structured links or they were unparseable. Falling back to generating links from top relevant results.")
            fallback_links = []
            
            # Take top N unique URLs from relevant results as a fallback (avoiding duplicates)
            for res in relevant_results[:MAX_RESULTS]: # Use MAX_RESULTS for fallback
                url = res.get("url")
                if url and url.startswith("http") and url not in unique_parsed_urls: # Ensure uniqueness
                    # Create a brief snippet from the content for the link text
                    snippet = res["content"][:100] + "..." if len(res["content"]) > 100 else res["content"]
                    fallback_links.append({"url": url, "text": snippet})
                    unique_parsed_urls.add(url)
            parsed_links.extend(fallback_links) # Extend, don't overwrite, to keep any valid parsed links
            
        logger.info(f"Parsed LLM response: Answer length={len(answer)}, Found {len(parsed_links)} sources after cross-referencing/fallback.")
        return {"answer": answer, "links": parsed_links}
    except Exception as e:
        error_msg = f"Error parsing LLM response: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        # Return a basic response structure with the error
        return {
            "answer": "Error: Could not fully parse the response from the language model. Please check the logs for details. Original response might be available without proper formatting.",
            "links": []
        }

# Define API routes
@app.post("/query", response_model=QueryResponse) # Specify response model for OpenAPI docs
async def query_knowledge_base(request: QueryRequest):
    """
    Main API endpoint to query the RAG knowledge base.
    Accepts a question and an optional base64 encoded image.
    Retrieves relevant chunks, enriches context, and generates an answer using an LLM.
    """
    try:
        # Log the incoming request
        logger.info(f"Received query request: question='{request.question[:100]}...', image_provided={request.image is not None}")
        
        if not API_KEY:
            error_msg = "AIPIPE_TOKEN environment variable not set. Please configure it."
            logger.error(error_msg)
            return JSONResponse(
                status_code=500,
                content={"answer": "Error: Server is not configured with an API key.", "links": []}
            )
            
        conn = None # Initialize conn to None
        try:
            conn = get_db_connection()
            
            # Process the query (handle text and optional image)
            logger.info("Processing query and generating embedding.")
            query_embedding = await process_multimodal_query(
                request.question,
                request.image
            )
            
            # Find similar content
            logger.info("Finding similar content in the knowledge base.")
            relevant_results = await find_similar_content(query_embedding, conn)
            
            if not relevant_results:
                logger.info("No relevant results found in knowledge base for the query.")
                return QueryResponse(
                    answer="I couldn't find any relevant information in my knowledge base to answer this question.",
                    links=[]
                )
            
            # Enrich results with adjacent chunks for better context
            logger.info("Enriching relevant results with adjacent chunks.")
            enriched_results = await enrich_with_adjacent_chunks(conn, relevant_results)
            
            # Generate answer using the LLM with the enriched context
            logger.info("Generating answer using LLM.")
            llm_response_text = await generate_answer(request.question, enriched_results)
            
            # Pass relevant_results to parse_llm_response for robust source extraction
            logger.info("Parsing LLM response to extract answer and sources.")
            parsed_response = parse_llm_response(llm_response_text, relevant_results)
            
            # Log the final result structure (without full content for brevity)
            logger.info(f"Returning final result: answer_length={len(parsed_response['answer'])}, num_links={len(parsed_response['links'])}")
            
            # Return the response in the exact format required by the QueryResponse Pydantic model
            return QueryResponse(
                answer=parsed_response["answer"],
                links=parsed_response["links"]
            )
        except HTTPException as he:
            # Re-raise HTTPExceptions directly, as they are already formatted for FastAPI
            logger.error(f"HTTPException during query processing: {he.detail}")
            raise he
        except Exception as e:
            # Catch any unexpected exceptions during the query processing pipeline
            error_msg = f"An unexpected error occurred during query processing: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={"answer": "An internal server error occurred while processing your request.", "links": []}
            )
        finally:
            # Ensure the database connection is closed
            if conn:
                conn.close()
                logger.info("Database connection closed.")
    except Exception as e:
        # Top-level catch for any exceptions before the main try-finally in the route
        error_msg = f"Unhandled exception in query_knowledge_base route handler: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"answer": "An unhandled server error occurred.", "links": []}
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Provides a health check endpoint for the API.
    Checks database connectivity and counts of chunks and embeddings.
    """
    try:
        # Try to connect to the database as part of health check
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if tables exist and have data
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
        discourse_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
        markdown_count = cursor.fetchone()[0]
        
        # Check if any embeddings exist
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks WHERE embedding IS NOT NULL")
        discourse_embeddings = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks WHERE embedding IS NOT NULL")
        markdown_embeddings = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "status": "healthy", 
            "database": "connected", 
            "api_key_set": bool(API_KEY),
            "discourse_chunks": discourse_count,
            "markdown_chunks": markdown_count,
            "discourse_embeddings": discourse_embeddings,
            "markdown_embeddings": markdown_embeddings
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e), "api_key_set": bool(API_KEY)}
        )

if __name__ == "__main__":
    # Runs the FastAPI application using Uvicorn.
    # reload=True is good for development; set to False for production.
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
