import asyncio
from functools import lru_cache
import json
import logging
import aiohttp
from litellm import AsyncOpenAI
import requests
from random import shuffle
from openai import OpenAI
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY =  os.getenv('API_KEY') 
MODEL = "gpt-3.5-turbo"
DATASET_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
MAX_QUERIES = 10
CACHE_SIZE = 1000
MAX_RETRIES = 3 

PROMPT_VARIATIONS = [
    "Categorize and answer the following question. Respond in JSON format with 'category' and 'answer' fields:",
    "Please provide a category and answer for this question. Use JSON format with 'category' and 'answer' keys:",
    "Analyze and respond to the following question. Return a JSON with 'category' for the question type and 'answer' for your response:",
]

async def download_dataset():
    logging.info("Downloading dataset...")
    async with aiohttp.ClientSession() as session:
        async with session.get(DATASET_URL) as response:
            data = await response.json()
    
    questions = [
        qa["question"]
        for article in data["data"]
        for paragraph in article["paragraphs"]
        for qa in paragraph["qas"]
    ]
    
    shuffle(questions)
    questions = questions[:MAX_QUERIES]
    logging.info(f"Processed {len(questions)} questions.")
    return questions

@lru_cache(maxsize=CACHE_SIZE)
async def get_response(prompt: str, client: AsyncOpenAI) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides answers in JSON format with 'category' and 'answer' fields."},
                    {"role": "user", "content": f"{PROMPT_VARIATIONS[attempt % len(PROMPT_VARIATIONS)]} {prompt}"}
                ],
                user="experiment_run_1"
            )
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            logging.warning(f"JSON parsing error on attempt {attempt + 1}, retrying...")
            await asyncio.sleep(2 ** attempt) 
        except Exception as e:
            logging.error(f"Error on attempt {attempt + 1}: {str(e)}")
            await asyncio.sleep(2 ** attempt)  
    
    logging.error(f"Failed to get response after {MAX_RETRIES} attempts")
    return {"category": "error", "answer": "Failed to get response"}

async def process_dataset(questions, client):
    results = []
    async def process_question(question):
        response = await get_response(question, client)
        return {
            "question": question,
            "category": response["category"],
            "answer": response["answer"]
        }
    
    tasks = [process_question(question) for question in questions]
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing questions"):
        result = await task
        results.append(result)
    
    return results

def analyze_results(results):
    categories = {}
    for result in results:
        category = result['category']
        categories[category] = categories.get(category, 0) + 1

    logging.info("Category distribution:")
    for category, count in categories.items():
        percentage = (count / len(results)) * 100
        logging.info(f"{category}: {percentage:.2f}%")

    most_common = max(categories, key=categories.get)
    logging.info(f"Most common category: {most_common}")

    avg_answer_length = sum(len(result['answer']) for result in results) / len(results)
    logging.info(f"Average answer length: {avg_answer_length:.2f} characters")

def save_results(results):
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    logging.info("Results saved to results.json")

async def main():
    logging.info("Starting experiment...")
    client = AsyncOpenAI(api_key=API_KEY)
    questions = await download_dataset()
    results = await process_dataset(questions, client)
    analyze_results(results)
    save_results(results)
    logging.info("Experiment complete.")

if __name__ == "__main__":
    asyncio.run(main())