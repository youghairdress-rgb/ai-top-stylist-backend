import os
import cv2
import numpy as np
import google.generativeai as genai
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import traceback
import json

# --- Configuration ---
# Set the Google API key from environment variables
try:
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Error reading or configuring Google API key: {e}")
    GOOGLE_API_KEY = None

# Initialize Flask App and CORS
app = Flask(__name__)
CORS(app)

# --- Health Check Endpoint ---
@app.route('/', methods=['GET'])
def health_check():
    """A simple endpoint to confirm the server is running."""
    return "Hello, Stylist AI is running!", 200

# --- AI and Image Processing Functions ---

def analyze_assets(front_image_bytes):
    """
    Analyzes all assets. In the future, this can be expanded.
    For now, it's a wrapper for dummy analysis functions.
    """
    print("Starting asset analysis...")
    # Dummy analysis functions
    face_skeleton_results = analyze_face_and_skeleton(front_image_bytes)
    personal_color_results = analyze_personal_color(front_image_bytes)
    full_diagnosis = {**face_skeleton_results, **personal_color_results}
    print("Asset analysis complete.")
    return full_diagnosis

def analyze_face_and_skeleton(image_bytes):
    print("-> Analyzing face and skeleton...")
    # This is a placeholder for actual Mediapipe logic
    return {
        "face_diagnosis": {"鼻": "丸みのある鼻", "口": "ふっくらした唇", "目": "丸い", "眉": "平行眉", "おでこ": "広め"},
        "skeleton_diagnosis": {"首の長さ": "普通", "顔の形": "丸顔", "ボディライン": "ストレート", "肩のライン": "なだらか"}
    }

def analyze_personal_color(image_bytes):
    print("-> Analyzing personal color...")
    # This is a placeholder for actual color analysis logic
    return {
        "personal_color_diagnosis": {"明度": "高", "ベースカラー": "イエローベース", "シーズン": "スプリング", "彩度": "中", "瞳の色": "ライトブラウン"}
    }

def call_llm_with_sdk(diagnosis_data):
    """
    Generates proposals by calling the Gemini API using the official Python SDK.
    """
    print("Calling LLM via official SDK...")
    if not GOOGLE_API_KEY:
        raise ValueError("Google API key is not configured.")

    # Using the stable 'gemini-pro' model with the official SDK
    model = genai.GenerativeModel('gemini-pro')
    
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
    print("\n--- Received request for /diagnose ---")
    if 'front_image' not in request.files:
        return jsonify({"error": "No front image provided"}), 400

    try:
        front_image_file = request.files['front_image']
        
        filestr = front_image_file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        # Resize image to prevent server overload
        max_width = 800
        if image.shape[1] > max_width:
            print(f"Image is large ({image.shape[1]}px width), resizing to {max_width}px.")
            scale = max_width / image.shape[1]
            new_height = int(image.shape[0] * scale)
            image = cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)
        
        _, image_bytes_for_analysis = cv2.imencode('.jpg', image)
        
        # Perform all analyses
        full_diagnosis = analyze_assets(image_bytes_for_analysis.tobytes())
        
        # Call LLM for proposals
        proposals_json_str = call_llm_with_sdk(full_diagnosis)
        proposals_data = json.loads(proposals_json_str)

        final_result = {"diagnosis": full_diagnosis, "proposals": proposals_data}
        
        print("Diagnosis process completed successfully.")
        return jsonify(final_result)

    except Exception as e:
        print(f"An error occurred during diagnosis: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route('/generate_style', methods=['POST'])
def generate_style():
    # Dummy implementation for virtual try-on
    return jsonify({"message": "Style generation is not fully implemented."})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

