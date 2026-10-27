import os
import time
import random
import requests
from google import genai
from dotenv import load_dotenv
from genai_client import get_vertex_client

load_dotenv()

print("🚀 Booting Resilient Ares RAG Pipeline (Vertex AI ADC)...")
COMPUTE_NODE_URL = "http://localhost:9000/search"
MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-3.5-flash-lite")


def execute_rag(user_question: str):
    print(f"\n👤 USER QUESTION: '{user_question}'")

    # --- STEP 1: RETRIEVAL ---
    print("🔍 [Retrieval] Querying Ares Coordinator / Compute Shards...")
    try:
        response = requests.post(COMPUTE_NODE_URL, json={"text": user_question, "top_k": 2, "use_hyde": True})
        if response.status_code != 200:
            print(f"❌ Error communicating with search endpoint: {response.text}")
            return
        search_data = response.json()
        retrieved_docs = [item['document'] for item in search_data.get('results', [])]
        context_string = "\n- ".join(retrieved_docs)
    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        return

    strict_prompt = f"""
    You are an expert answering questions. You must answer the user's question based ONLY on the provided Context. 
    If the answer is not contained in the Context, you must say "I do not have enough information to answer that."

    CONTEXT:
    - {context_string}

    USER QUESTION:
    {user_question}
    """

    # --- STEP 2: GENERATION (With Jitter Backoff) ---
    print("🧠 [Generation] Sending strict context to Vertex AI LLM...")
    client = get_vertex_client(strict_production=True)
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        try:
            llm_response = client.models.generate_content(
                model=MODEL_ID,
                contents=strict_prompt
            )

            print("\n=========================================")
            print("ARES AI RESPONSE:")
            print("=========================================")
            print(llm_response.text.strip())
            print("=========================================\n")
            return

        except Exception as e:
            print(f"⚠️ [Attempt {attempt}/{max_retries}] API Error: {e}")
            if attempt == max_retries:
                print("❌ Fatal Error: LLM Provider is down.")
                return

            sleep_time = (2 ** attempt) + random.uniform(0.1, 1.0)
            print(f"⏳ Sleeping for {sleep_time:.2f} seconds before retrying...")
            time.sleep(sleep_time)


if __name__ == "__main__":
    execute_rag("How plants be getting their energy?")