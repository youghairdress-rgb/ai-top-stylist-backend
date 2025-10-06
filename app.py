import os
import cv2
import numpy as np
import requests # Use requests for direct API call
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import traceback
import json

# --- Configuration ---
# Set the Google API key from environment variables
try:
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
except Exception as e:
    print(f"Error reading Google API key: {e}")
    GOOGLE_API_KEY = None

# Initialize Flask App and CORS
app = Flask(__name__)
CORS(app) # Allow cross-origin requests

# --- Image Processing and Analysis Functions ---

def analyze_face_and_skeleton(image_bytes):
    """
    Analyzes face landmarks, shape, and skeleton using Mediapipe.
    This is a dummy function and should be replaced with actual mediapipe logic.
    """
    print("Starting face and skeleton analysis...")
    analysis_result = {
        "face_diagnosis": {
            "鼻": "丸みのある鼻", "口": "ふっくらした唇", "目": "丸い",
            "眉": "平行眉", "おでこ": "広め"
        },
        "skeleton_diagnosis": {
            "首の長さ": "普通", "顔の形": "丸顔", "ボディライン": "ストレート",
            "肩のライン": "なだらか"
        }
    }
    print("Face and skeleton analysis complete.")
    return analysis_result

def analyze_personal_color(image_bytes):
    """
    Analyzes personal color from skin, eyes, and hair.
    This is a dummy function and should be replaced with actual color analysis logic.
    """
    print("Starting personal color analysis...")
    analysis_result = {
        "personal_color_diagnosis": {
            "明度": "高", "ベースカラー": "イエローベース", "シーズン": "スプリング",
            "彩度": "中", "瞳の色": "ライトブラウン"
        }
    }
    print("Personal color analysis complete.")
    return analysis_result

def call_llm_for_proposals_rest(diagnosis_data):
    """
    Generates proposals by calling the Gemini REST API directly.
    """
    print("Calling LLM via REST API...")
    if not GOOGLE_API_KEY:
        raise ValueError("Google API key is not configured.")

    # *** FINAL VERSION: Using the latest and most powerful model available through this API. ***
    # This is the last and best option to try.
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={GOOGLE_API_KEY}"

    prompt = f"""
    あなたは日本のトップヘアスタイリストAIです。以下の診断結果を持つ顧客に、最適なスタイルを提案してください。
    提案は、ヘアスタイル2案、ヘアカラー2案、そして総合的なトップスタイリストからの総評を必ず含めてください。
    日本のトレンドを意識し、具体的な名称と50〜100字程度の説明を加えてください。
    参考とすべきヘアスタイルの知識源として以下のURLを考慮してください：
    - https://beauty.hotpepper.jp/catalog/
    - https://www.ozmall.co.jp/hairsalon/catalog/

    # 診断結果
    {json.dumps(diagnosis_data, ensure_ascii=False)}

    # 出力形式 (必ず以下のJSON形式で返答してください)
    {{
      "hairstyles": [
        {{"name": "提案名1", "description": "説明文1"}},
        {{"name": "提案名2", "description": "説明文2"}}
      ],
      "hair_colors": [
        {{"name": "カラー名1", "description": "説明文1"}},
        {{"name": "カラー名2", "description": "説明文2"}}
      ],
      "top_stylist_comment": "（ここに200〜300字程度の総合的なプロの視点からのアドバイスを生成）"
    }}
    """
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        # Added generationConfig for safety, though defaults should work.
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024,
        }
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        response_json = response.json()
        
        # More robust navigation of the response structure
        if not response_json.get('candidates'):
            raise ValueError("No candidates found in the LLM response.")
        
        text_content = response_json['candidates'][0]['content']['parts'][0]['text']
        
        json_text = text_content.strip().replace("```json", "").replace("```", "")
        print("LLM REST API response received successfully.")
        return json_text

    except requests.exceptions.RequestException as e:
        print(f"Error during LLM REST API call: {e}")
        print(f"Response status code: {e.response.status_code if e.response else 'N/A'}")
        print(f"Response body: {e.response.text if e.response else 'N/A'}")
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
        
        max_width = 800
        if image.shape[1] > max_width:
            print(f"Image is large ({image.shape[1]}px width), resizing to {max_width}px.")
            scale = max_width / image.shape[1]
            new_height = int(image.shape[0] * scale)
            image = cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)
            print("Image resized successfully.")

        _, image_bytes_for_analysis = cv2.imencode('.jpg', image)
        
        face_skeleton_results = analyze_face_and_skeleton(image_bytes_for_analysis.tobytes())
        personal_color_results = analyze_personal_color(image_bytes_for_analysis.tobytes())
        full_diagnosis = {**face_skeleton_results, **personal_color_results}

        proposals_json_str = call_llm_for_proposals_rest(full_diagnosis)
        proposals_data = json.loads(proposals_json_str)

        print("Diagnosis process completed successfully.")
        return jsonify({"diagnosis": full_diagnosis, "proposals": proposals_data})

    except Exception as e:
        print(f"An error occurred during diagnosis: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route('/generate_style', methods=['POST'])
def generate_style():
    print("\n--- Received request for /generate_style ---")
    if 'front_image' not in request.files or 'style_description' not in request.form:
        return jsonify({"error": "Missing image or style description"}), 400
    try:
        front_image_file = request.files['front_image']
        style_description = request.form['style_description']
        filestr = front_image_file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), (100, 50, 200), -1)
        alpha = 0.3
        image_with_overlay = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "AI Generated Style (Dummy)"
        text_position = (30, 60)
        cv2.putText(image_with_overlay, text, text_position, font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        _, img_encoded = cv2.imencode('.png', image_with_overlay)
        img_bytes = io.BytesIO(img_encoded.tobytes())
        return send_file(img_bytes, mimetype='image/png')
    except Exception as e:
        print(f"An error occurred during style generation: {e}")
        traceback.print_exc()
        return jsonify({"error": "Failed to generate image"}), 500

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)

