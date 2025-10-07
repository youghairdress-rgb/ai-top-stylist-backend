import os
import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import traceback
import json
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
# Set the Google API key from environment variables
try:
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
except Exception as e:
    print(f"Error reading Google API key: {e}")
    GOOGLE_API_KEY = None

# Initialize Flask App and CORS
app = Flask(__name__)
CORS(app)

# In-memory storage for task status and results.
# In a production app, use Redis or a database instead.
tasks = {}

# Use a thread pool to manage background tasks
executor = ThreadPoolExecutor(max_workers=5)


# --- Health Check Endpoint ---
@app.route('/', methods=['GET'])
def health_check():
    """A simple endpoint to confirm the server is running."""
    return "Hello, Stylist AI is running!", 200


# --- AI and Image Processing Functions (to be run in background) ---

def run_full_diagnosis(task_id, image_bytes):
    """The main function that performs all heavy lifting."""
    try:
        print(f"[Task {task_id}] Starting full diagnosis in background.")
        
        # 1. Analyze face and skeleton (dummy)
        face_skeleton_results = analyze_face_and_skeleton(image_bytes)
        tasks[task_id]['status'] = 'analyzing_color'

        # 2. Analyze personal color (dummy)
        personal_color_results = analyze_personal_color(image_bytes)
        tasks[task_id]['status'] = 'calling_llm'
        
        full_diagnosis = {**face_skeleton_results, **personal_color_results}

        # 3. Call LLM for proposals
        proposals_json_str = call_llm_for_proposals_rest(full_diagnosis)
        proposals_data = json.loads(proposals_json_str)

        # 4. Store the final result and mark as complete
        final_result = {"diagnosis": full_diagnosis, "proposals": proposals_data}
        tasks[task_id]['result'] = final_result
        tasks[task_id]['status'] = 'complete'
        print(f"[Task {task_id}] Diagnosis complete and result stored.")

    except Exception as e:
        print(f"[Task {task_id}] CRITICAL ERROR in background thread: {e}")
        traceback.print_exc()
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)


def analyze_face_and_skeleton(image_bytes):
    print("Starting face and skeleton analysis...")
    # This is a dummy function.
    return {
        "face_diagnosis": {"鼻": "丸みのある鼻", "口": "ふっくらした唇", "目": "丸い", "眉": "平行眉", "おでこ": "広め"},
        "skeleton_diagnosis": {"首の長さ": "普通", "顔の形": "丸顔", "ボディライン": "ストレート", "肩のライン": "なだらか"}
    }

def analyze_personal_color(image_bytes):
    print("Starting personal color analysis...")
    # This is a dummy function.
    return {
        "personal_color_diagnosis": {"明度": "高", "ベースカラー": "イエローベース", "シーズン": "スプリング", "彩度": "中", "瞳の色": "ライトブラウン"}
    }

def call_llm_for_proposals_rest(diagnosis_data):
    print("Calling LLM via REST API...")
    if not GOOGLE_API_KEY:
        raise ValueError("Google API key is not configured.")

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GOOGLE_API_KEY}"
    prompt = f"""
    あなたは日本のトップヘアスタイリストAIです。以下の診断結果を持つ顧客に、最適なスタイルを提案してください。
    提案は、ヘアスタイル2案、ヘアカラー2案、そして総合的なトップスタイリストからの総評を必ず含めてください。
    日本のトレンドを意識し、具体的な名称と50〜100字程度の説明を加えてください。
    # 診断結果: {json.dumps(diagnosis_data, ensure_ascii=False)}
    # 出力形式: 必ず以下のJSON形式で返答してください
    {{
      "hairstyles": [{{"name": "提案名1", "description": "説明文1"}}, {{"name": "提案名2", "description": "説明文2"}}],
      "hair_colors": [{{"name": "カラー名1", "description": "説明文1"}}, {{"name": "カラー名2", "description": "説明文2"}}],
      "top_stylist_comment": "（ここに200〜300字程度の総合的なプロの視点からのアドバイスを生成）"
    }}
    """
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=100) # Add a client-side timeout
        response.raise_for_status()
        response_json = response.json()
        text_content = response_json['candidates'][0]['content']['parts'][0]['text']
        json_text = text_content.strip().replace("```json", "").replace("```", "")
        print("LLM REST API response received successfully.")
        return json_text
    except requests.exceptions.RequestException as e:
        print(f"CRITICAL ERROR during LLM REST API call: {e}")
        traceback.print_exc()
        raise

# --- API Endpoints ---

@app.route('/diagnose', methods=['POST'])
def diagnose():
    print("\n--- Received request for /diagnose ---")
    if 'front_image' not in request.files:
        return jsonify({"error": "No front image provided"}), 400

    front_image_file = request.files['front_image']
    filestr = front_image_file.read()

    # Create a unique ID for this diagnosis task
    task_id = str(uuid.uuid4())
    tasks[task_id] = {'status': 'processing_image'}
    
    print(f"Created new task with ID: {task_id}")

    # Start the long-running diagnosis in a background thread
    executor.submit(run_full_diagnosis, task_id, filestr)

    # Immediately return the task ID to the client
    return jsonify({"status": "pending", "task_id": task_id}), 202

@app.route('/get_result', methods=['GET'])
def get_result():
    task_id = request.args.get('task_id')
    if not task_id or task_id not in tasks:
        return jsonify({"error": "Invalid or missing task_id"}), 404
    
    task = tasks[task_id]
    print(f"Polling for task {task_id}, current status: {task['status']}")

    if task['status'] == 'complete':
        # Return the final result and clean up the task
        result = task.get('result')
        del tasks[task_id] 
        return jsonify({"status": "complete", "data": result})
    elif task['status'] == 'error':
        error_message = task.get('error', 'An unknown error occurred.')
        del tasks[task_id]
        return jsonify({"status": "error", "message": error_message}), 500
    else:
        # The task is still running
        return jsonify({"status": "pending"})

# Dummy endpoint, unchanged
@app.route('/generate_style', methods=['POST'])
def generate_style():
    # ... (code is unchanged)
    return jsonify({"message": "This endpoint is not implemented in the async example."})


if __name__ == '__main__':
    app.run(debug=True, port=5001)

