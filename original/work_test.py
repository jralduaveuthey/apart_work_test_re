import json
import time
import requests
from random import shuffle
from openai import OpenAI

API_KEY = "sk-yourapikeyhere"
MODEL = "gpt-3.5-turbo"
DATASET_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
MAX_QUERIES = 10000
PROMPT_VARIATIONS = [
    "Categorize and answer the following question. Respond in JSON format with 'category' and 'answer' fields:",
    "Please provide a category and answer for this question. Use JSON format with 'category' and 'answer' keys:",
    "Analyze and respond to the following question. Return a JSON with 'category' for the question type and 'answer' for your response:",
]
SELECTED_PROMPT = PROMPT_VARIATIONS[1]

def download_dataset():
    print("Downloading dataset...")
    response = requests.get(DATASET_URL)
    data = response.json()
    questions = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                questions.append(qa["question"])
    shuffle(questions)
    questions = questions[:MAX_QUERIES]
    print(f"Downloaded {len(questions)} questions.")
    return questions

def get_response(prompt, client):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides answers in JSON format."},
                {"role": "user", "content": f"{SELECTED_PROMPT} {prompt}"}
            ],
            user="experiment_run_1"
        )
        return json.loads(response.choices[0].message.content)
    except json.decoder.JSONDecodeError:
        print("JSON parsing error, waiting for 10 seconds...")
        time.sleep(10)
        return get_response(prompt, client)

def process_dataset(questions, client):
    results = []
    for i, question in enumerate(questions):
        if i % 100 == 0:
            print(f"Processing question {i+1}/{len(questions)}")
        response = get_response(question, client)
        results.append({
            "question": question,
            "category": response["category"],
            "answer": response["answer"]
        })
    return results

def analyze_results(results):
    categories = {}
    for result in results:
        category = result['category']
        categories[category] = categories.get(category, 0) + 1
    
    print("Category distribution:")
    for category, count in categories.items():
        percentage = (count / len(results)) * 100
        print(f"{category}: {percentage:.2f}%")
    
    most_common = max(categories, key=categories.get)
    print(f"Most common category: {most_common}")
    
    avg_answer_length = sum(len(result['answer']) for result in results) / len(results)
    print(f"Average answer length: {avg_answer_length:.2f} characters")

def save_results(results):
    with open("results.json", "w") as f:
        json.dump(results, f)

def main():
    print("Starting experiment...")
    client = OpenAI(api_key=API_KEY)
    questions = download_dataset()
    results = process_dataset(questions, client)
    analyze_results(results)
    save_results(results)
    print("Experiment complete.")

if __name__ == "__main__":
    main()