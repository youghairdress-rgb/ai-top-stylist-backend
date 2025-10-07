import os
import cv2
import numpy as np
import google.generativeai as genai
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import traceback
import json
import uuid
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
tasks = {}
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
        tasks[task_id]['status'] = 'analyzing_assets'
        
        # Configure the SDK here, inside the thread, to be safe.
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
        
        # Dummy analysis functions
        face_skeleton_results = analyze_face_and_skeleton(image_bytes)
        personal_color_results = analyze_personal_color(image_bytes)
        full_diagnosis = {**face_skeleton_results, **personal_color_results}
        
        tasks[task_id]['status'] = 'calling_llm'
        
        # Call LLM for proposals
        proposals_json_str = call_llm_with_sdk(full_diagnosis)
        proposals_data = json.loads(proposals_json_str)

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
    print("-> Analyzing face and skeleton...")
    return {
        "face_diagnosis": {"鼻": "丸みのある鼻", "口": "ふっくらした唇", "目": "丸い", "眉": "平行眉", "おでこ": "広め"},
        "skeleton_diagnosis": {"首の長さ": "普通", "顔の形": "丸顔", "ボディライン": "ストレート", "肩のライン": "なだらか"}
    }

def analyze_personal_color(image_bytes):
    print("-> Analyzing personal color...")
    return {
        "personal_color_diagnosis": {"明度": "高", "ベースカラー": "イエローベース", "シーズン": "スプリング", "彩度": "中", "瞳の色": "ライトブラウン"}
    }

def call_llm_with_sdk(diagnosis_data):
    print("Calling LLM via official SDK...")
    if not GOOGLE_API_KEY:
        raise ValueError("Google API key is not configured.")

    # Using the specified 'gemini-1.5-pro' model with the latest SDK
    model = genai.GenerativeModel('gemini-1.5-pro')
    
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
    
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        print("LLM SDK response received successfully.")
        return json_text
    except Exception as e:
        print(f"CRITICAL ERROR during LLM SDK call: {e}")
        traceback.print_exc()
        raise

# --- API Endpoints ---
@app.route('/diagnose', methods=['POST'])
def diagnose():
    print("\n--- Received request to start diagnosis ---")
    if 'front_image' not in request.files:
        return jsonify({"error": "No front image provided"}), 400

    filestr = request.files['front_image'].read()
    task_id = str(uuid.uuid4())
    tasks[task_id] = {'status': 'pending'}
    
    print(f"Created new task with ID: {task_id}")
    executor.submit(run_full_diagnosis, task_id, filestr)
    
    return jsonify({"status": "pending", "task_id": task_id}), 202

@app.route('/get_result', methods=['GET'])
def get_result():
    task_id = request.args.get('task_id')
    if not task_id or task_id not in tasks:
        return jsonify({"error": "Invalid or missing task_id"}), 404
    
    task = tasks.get(task_id, {})
    status = task.get('status', 'not_found')
    
    print(f"Polling for task {task_id}, current status: {status}")

    if status == 'complete':
        result = task.get('result')
        tasks.pop(task_id, None) 
        return jsonify({"status": "complete", "data": result})
    elif status == 'error':
        error_message = task.get('error', 'An unknown error occurred.')
        tasks.pop(task_id, None)
        return jsonify({"status": "error", "message": error_message}), 500
    else:
        return jsonify({"status": status})

@app.route('/generate_style', methods=['POST'])
def generate_style():
    # Dummy implementation
    return jsonify({"message": "Style generation is not fully implemented."})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

