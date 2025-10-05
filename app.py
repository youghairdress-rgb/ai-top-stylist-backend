import os
import cv2
import numpy as np
import google.generativeai as genai
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import io
import traceback

# --- Configuration ---
# Set the Google API key from environment variables
try:
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Error configuring Google API: {e}")
    # Handle the case where the API key is not set
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
    # Dummy data simulating analysis results
    analysis_result = {
        "face_diagnosis": {
            "鼻": "丸みのある鼻",
            "口": "ふっくらした唇",
            "目": "丸い",
            "眉": "平行眉",
            "おでこ": "広め"
        },
        "skeleton_diagnosis": {
            "首の長さ": "普通",
            "顔の形": "丸顔",
            "ボディライン": "ストレート",
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
    # Dummy data simulating analysis results
    analysis_result = {
        "personal_color_diagnosis": {
            "明度": "高",
            "ベースカラー": "イエローベース",
            "シーズン": "スプリング",
            "彩度": "中",
            "瞳の色": "ライトブラウン"
        }
    }
    print("Personal color analysis complete.")
    return analysis_result

def call_llm_for_proposals(diagnosis_data):
    """
    Generates hairstyle, color, and other proposals using the Gemini LLM.
    """
    print("Calling LLM for proposals...")
    if not GOOGLE_API_KEY:
        raise ValueError("Google API key is not configured.")

    model = genai.GenerativeModel('gemini-1.5-flash')

    # Constructing the prompt from diagnosis data
    prompt = f"""
    あなたは日本のトップヘアスタイリストAIです。以下の診断結果を持つ顧客に、最適なスタイルを提案してください。
    提案は、ヘアスタイル2案、ヘアカラー2案、そして総合的なトップスタイリストからの総評を必ず含めてください。
    日本のトレンドを意識し、具体的な名称と50〜100字程度の説明を加えてください。
    参考とすべきヘアスタイルの知識源として以下のURLを考慮してください：
    - https://beauty.hotpepper.jp/catalog/
    - https://www.ozmall.co.jp/hairsalon/catalog/

    # 診断結果
    {diagnosis_data}

    # 出力形式 (JSON)
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
    
    try:
        response = model.generate_content(prompt)
        # Extracting the JSON part from the response text
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        print("LLM response received.")
        return json_text
    except Exception as e:
        print(f"Error during LLM call: {e}")
        traceback.print_exc()
        raise

# --- API Endpoints ---

@app.route('/diagnose', methods=['POST'])
def diagnose():
    print("\n--- Received request for /diagnose ---")
    if 'front_image' not in request.files:
        print("Error: 'front_image' not in request files.")
        return jsonify({"error": "No front image provided"}), 400

    try:
        # Read image file from request
        front_image_file = request.files['front_image']
        
        # Read the image data into a numpy array
        filestr = front_image_file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        # *** KEY IMPROVEMENT: Resize the image to speed up processing ***
        max_width = 800
        if image.shape[1] > max_width:
            print(f"Image is large ({image.shape[1]}px width), resizing to {max_width}px.")
            scale = max_width / image.shape[1]
            new_height = int(image.shape[0] * scale)
            image = cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)
            print("Image resized successfully.")

        # Convert the resized image back to bytes for analysis functions
        _, image_bytes_for_analysis = cv2.imencode('.jpg', image)
        image_bytes_for_analysis = image_bytes_for_analysis.tobytes()

        # Perform analyses
        face_skeleton_results = analyze_face_and_skeleton(image_bytes_for_analysis)
        personal_color_results = analyze_personal_color(image_bytes_for_analysis)

        # Combine diagnosis results
        full_diagnosis = {**face_skeleton_results, **personal_color_results}

        # Get proposals from LLM
        proposals_json_str = call_llm_for_proposals(full_diagnosis)
        
        # In a real application, you would parse and validate the JSON
        import json
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
        print(f"Style description: {style_description}")

        # --- Dummy Image Generation Logic ---
        # This part should be replaced with a real call to an image generation AI (e.g., Stable Diffusion Inpainting)
        print("Generating dummy image...")
        filestr = front_image_file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Add a semi-transparent overlay to simulate a style change
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), (100, 50, 200), -1) # BGR color
        alpha = 0.3
        image_with_overlay = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
        # Add text to the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "AI Generated Style (Dummy)"
        text_position = (30, 60)
        cv2.putText(image_with_overlay, text, text_position, font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

        # Convert the final image to a byte stream to send back
        _, img_encoded = cv2.imencode('.png', image_with_overlay)
        img_bytes = io.BytesIO(img_encoded.tobytes())
        
        print("Dummy image generated and sent.")
        return send_file(img_bytes, mimetype='image/png')

    except Exception as e:
        print(f"An error occurred during style generation: {e}")
        traceback.print_exc()
        return jsonify({"error": "Failed to generate image"}), 500

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)

