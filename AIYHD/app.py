import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
import io
import os
import google.generativeai as genai

# --- Flaskアプリケーションの初期化 ---
app = Flask(__name__)
CORS(app)  # CORSを有効にし、別ドメインのフロントエンドからのリクエストを許可

# --- Google AI (Gemini) APIキーの設定 ---
# 環境変数からAPIキーを読み込みます。デプロイ時にこの環境変数を設定します。
try:
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("APIキーが設定されていません。環境変数 'GOOGLE_API_KEY' を設定してください。")
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"APIキーの設定中にエラーが発生しました: {e}")


# --- MediaPipeの初期化 ---
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# --- 画像・動画解析のヘルパー関数 ---

def analyze_face_skeleton(image_bytes):
    """
    画像データから顔・骨格診断を行う（ダミー実装）。
    MediaPipeを使用して顔のランドマークとポーズを検出する。
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(img_rgb)
    pose_results = pose.process(img_rgb)

    diagnostics = {
        "face_diagnosis": {
            "鼻": "丸みのある鼻", "口": "ふっくらした唇", "目": "丸い",
            "眉": "平行眉", "おでこ": "広め"
        },
        "skeleton_diagnosis": {
            "首の長さ": "普通", "顔の形": "丸顔", "ボディライン": "ストレート",
            "肩のライン": "なだらか"
        }
    }
    if not face_results.multi_face_landmarks:
        diagnostics["face_diagnosis"]["error"] = "顔を検出できませんでした。"
    if not pose_results.pose_landmarks:
        diagnostics["skeleton_diagnosis"]["error"] = "体を検出できませんでした。"
    return diagnostics

def analyze_personal_color(image_bytes):
    """
    画像データからパーソナルカラー診断を行う（ダミー実装）。
    """
    return {
        "personal_color_diagnosis": {
            "明度": "高", "ベースカラー": "イエローベース", "シーズン": "スプリング",
            "彩度": "中", "瞳の色": "ライトブラウン"
        }
    }

def analyze_hair_from_video(video_bytes):
    """
    動画データから毛髪診断を行う（ダミー実装）。
    """
    return {
        "hair_diagnosis": {
            "クセ": "軽いくせ毛", "ボリューム感": "普通", "明度": "やや明るめ",
            "損傷度合い": "中"
        }
    }

# --- AI（LLM・画像生成）連携のヘルパー関数 ---

def generate_llm_prompt(diagnostics):
    """
    診断結果を基に、LLMへの指示プロンプトを生成する。
    """
    return f"""
あなたはカリスマ的なトップヘアスタイリストAIです。以下の診断結果を持つ顧客に対して、最高のスタイルを提案してください。

# 顧客の診断結果
{json.dumps(diagnostics, indent=2, ensure_ascii=False)}

# 指示
上記の診断結果を総合的に分析し、以下のフォーマットで提案を生成してください。
- 提案はプロフェッショナルかつ、顧客を勇気づけるようなポジティブなトーンで記述してください。
- 各提案のネーミングは魅力的で、説明は50〜100字程度にまとめてください。
- ヘアスタイルとヘアカラーの提案は、以下のWebサイトのトレンドを参考にしてください。
  - https://beauty.hotpepper.jp/catalog/
  - https://www.ozmall.co.jp/hairsalon/catalog/
- 必ず指定されたJSON形式で出力してください。

# 出力フォーマット（JSON形式）
{{
  "hairstyle_proposal": [
    {{"name": "提案ヘアスタイル名1", "description": "ヘアスタイルの説明文1"}},
    {{"name": "提案ヘアスタイル名2", "description": "ヘアスタイルの説明文2"}}
  ],
  "haircolor_proposal": [
    {{"name": "提案ヘアカラー名1", "description": "ヘアカラーの説明文1"}},
    {{"name": "提案ヘアカラー名2", "description": "ヘアカラーの説明文2"}}
  ],
  "makeup_proposal": {{
    "lip_color": "リップカラー", "eyeshadow_color": "アイシャドウ", "cheek_color": "チーク", "foundation": "ファンデーション"
  }},
  "fashion_proposal": {{
    "base_color": "ベースカラー", "accent_color": "差し色", "material": "素材", "silhouette": "シルエット"
  }},
  "final_comment": "トップスタイリストとしての総合的なコメント（200〜300字）"
}}
"""

def call_llm_api(prompt):
    """
    【更新】Gemini APIを呼び出し、提案を取得する。
    """
    print("--- Calling Gemini API ---")
    try:
        # 使用するモデルを設定
        # テキスト生成とJSON出力に強い 'gemini-1.5-flash-latest' を使用
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        # AIからのレスポンスが必ずJSON形式になるように設定
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json"
        )
        
        response = model.generate_content(prompt, generation_config=generation_config)
        
        # AIからのレスポンス（JSONテキスト）をPythonの辞書オブジェクトに変換
        return json.loads(response.text)

    except Exception as e:
        print(f"Gemini API呼び出しエラー: {e}")
        # エラーが発生した場合は、以前のダミーレスポンスを返す
        return {
          "hairstyle_proposal": [{"name": "エラー", "description": "AIとの通信に失敗しました。"}],
          "haircolor_proposal": [{"name": "エラー", "description": "AIとの通信に失敗しました。"}],
          "makeup_proposal": {}, "fashion_proposal": {},
          "final_comment": "AIとの通信中にエラーが発生しました。しばらくしてから再度お試しください。"
        }


def generate_dummy_image(face_image_bytes, style_text, adjustment_params):
    """
    顔写真とスタイル指示に基づき、合成画像を生成する（ダミー実装）。
    将来的にはこの関数を実際の画像生成AI（Imagenなど）に置き換えます。
    """
    print(f"画像生成リクエスト: {style_text}, 調整: {adjustment_params}")
    nparr = np.frombuffer(face_image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    text_to_draw = f"Style: {style_text}"
    if adjustment_params.get('text_prompt'):
        text_to_draw += f" ({adjustment_params['text_prompt']})"
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (30, 80)
    font_scale = 1
    color = (255, 255, 255)
    thickness = 2
    
    text_size, _ = cv2.getTextSize(text_to_draw, font, font_scale, thickness)
    cv2.rectangle(img, (position[0]-5, position[1] - text_size[1] - 10), (position[0] + text_size[0] + 5, position[1] + 10), (0,0,0), -1)
    cv2.putText(img, text_to_draw, position, font, font_scale, color, thickness, cv2.LINE_AA)
    
    is_success, buffer = cv2.imencode(".png", img)
    return io.BytesIO(buffer) if is_success else None


# --- APIエンドポイントの定義 ---

@app.route('/diagnose', methods=['POST'])
def diagnose():
    if 'front_image' not in request.files:
        return jsonify({"error": "正面画像が見つかりません"}), 400

    front_image = request.files['front_image'].read()
    
    try:
        face_skeleton_result = analyze_face_skeleton(front_image)
        personal_color_result = analyze_personal_color(front_image)
        hair_result = {"hair_diagnosis": {"クセ": "軽いくせ毛","ボリューム感": "普通","明度": "やや明るめ","損傷度合い": "中"}}

        all_diagnostics = {**face_skeleton_result, **personal_color_result, **hair_result}
        llm_prompt = generate_llm_prompt(all_diagnostics)
        ai_proposal = call_llm_api(llm_prompt)

        return jsonify({"diagnostics": all_diagnostics, "proposal": ai_proposal})
    except Exception as e:
        app.logger.error(f"診断エラー: {e}")
        return jsonify({"error": "サーバーでエラーが発生しました"}), 500

@app.route('/generate_style', methods=['POST'])
def generate_style():
    if 'front_image' not in request.files or 'style_text' not in request.form:
        return jsonify({"error": "必須データが見つかりません"}), 400

    front_image = request.files['front_image'].read()
    style_text = request.form['style_text']
    adjustment_params = {"text_prompt": request.form.get('text_prompt')}

    try:
        generated_image_io = generate_dummy_image(front_image, style_text, adjustment_params)
        if generated_image_io is None:
             return jsonify({"error": "画像生成に失敗しました"}), 500
        generated_image_io.seek(0)
        return send_file(generated_image_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"画像生成エラー: {e}")
        return jsonify({"error": "サーバーでエラーが発生しました"}), 500

# --- アプリケーションの実行 ---
if __name__ == '__main__':
    # Flaskサーバーを起動します。debug=Trueは開発用です。
    app.run(debug=True, port=5001)

