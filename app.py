import os
import cv2
import numpy as np
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import traceback

# --- 設定 ---
# 環境変数からAPIキーを安全に読み込む
# サーバー起動時にキーがない場合はエラーを出力して終了させ、問題を早期に検知します。
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("環境変数 'GOOGLE_API_KEY' が設定されていません。デプロイ環境で設定してください。")
genai.configure(api_key=GOOGLE_API_KEY)

# FlaskアプリとCORSの初期化
app = Flask(__name__)
# 本番環境では、セキュリティ向上のため、実際のフロントエンドのドメインに限定することを推奨します。
# 例: CORS(app, resources={r"/*": {"origins": "https://your-liff-app.web.app"}})
CORS(app)

# --- ヘルスチェック用エンドポイント ---
@app.route('/', methods=['GET'])
def health_check():
    """サーバーが正常に起動しているかを確認するためのエンドポイント"""
    print("Health check endpoint was hit.")
    return "AI Top Stylist API is running!", 200

# --- AI・画像処理関数 ---

def analyze_assets_dummy(image_bytes):
    """
    【ダミー関数】画像から顔・骨格・パーソナルカラーを分析します。
    TODO: ここにMediapipeを使用した実際の分析ロジックを実装します。
    """
    print("-> (Dummy) Analyzing user assets...")
    # 提供された資料に基づいたダミーデータを返却します。
    return {
        "face_diagnosis": {"鼻": "丸みのある鼻", "口": "ふっくらした唇", "目": "丸い", "眉": "平行眉", "おでこ": "広め"},
        "skeleton_diagnosis": {"首の長さ": "普通", "顔の形": "丸顔", "ボディライン": "ストレート", "肩のライン": "なだらか"},
        "personal_color_diagnosis": {"明度": "高", "ベースカラー": "イエローベース", "シーズン": "スプリング", "彩度": "中", "瞳の色": "ライトブラウン"}
    }

def get_style_proposals_from_llm(diagnosis_data):
    """
    診断結果を基に、Google Geminiモデルを呼び出してスタイル提案を生成します。
    """
    print("-> Calling Gemini API to get style proposals...")
    
    # 最新の安定版モデルを使用します。
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    
    prompt = f"""
    あなたは日本のトップヘアスタイリストAIです。以下の診断結果を持つお客様に、最適なスタイルを提案してください。
    提案は、ヘアスタイル2案、ヘアカラー2案、そして総合的なトップスタイリストからの総評を必ず含めてください。
    日本の20代〜30代女性のトレンドを強く意識し、具体的なヘアスタイルの名称と50〜100字程度の説明を加えてください。

    # お客様の診断結果: 
    {json.dumps(diagnosis_data, ensure_ascii=False, indent=2)}

    # 出力形式: 
    必ず以下のキーを持つJSON形式で、```json ```ブロック内に記述して返答してください。
    {{
      "hairstyles": [
        {{"name": "提案ヘアスタイル名1", "description": "ヘアスタイル1の具体的な説明"}}, 
        {{"name": "提案ヘアスタイル名2", "description": "ヘアスタイル2の具体的な説明"}}
      ],
      "hair_colors": [
        {{"name": "提案ヘアカラー名1", "description": "ヘアカラー1の具体的な説明"}}, 
        {{"name": "提案ヘアカラー名2", "description": "ヘアカラー2の具体的な説明"}}
      ],
      "top_stylist_comment": "（ここに200〜300字程度の、プロの視点からの総合的なアドバイスを生成してください）"
    }}
    """
    
    try:
        # API呼び出しにタイムアウトを設定し、長時間待機を防ぎます。
        response = model.generate_content(prompt, request_options={"timeout": 120})
        
        # レスポンスからJSON部分のみを安全に抽出します。
        json_text = response.text.strip().lstrip("```json").rstrip("```")
        print("-> Gemini API response received successfully.")
        return json.loads(json_text)
        
    except Exception as e:
        print(f"CRITICAL ERROR during Gemini API call: {e}")
        traceback.print_exc()
        # エラーが発生した場合、フロントエンドに問題を伝えるために例外を再発生させます。
        raise ValueError(f"Gemini APIとの通信に失敗しました: {e}")

# --- APIエンドポイント ---
@app.route('/diagnose', methods=['POST'])
def diagnose():
    print("\n--- Received request for /diagnose ---")
    if 'front_image' not in request.files:
        print("ERROR: No 'front_image' provided in the request.")
        return jsonify({"error": "正面の画像ファイルが見つかりません。"}), 400

    try:
        front_image_file = request.files['front_image']
        
        # 画像データをメモリ上で処理
        image_stream = front_image_file.read()
        np_image = np.frombuffer(image_stream, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        
        # ダミーの分析処理を呼び出し
        full_diagnosis = analyze_assets_dummy(image_stream)

        # LLMを呼び出して提案を取得
        proposals_data = get_style_proposals_from_llm(full_diagnosis)

        # 最終結果を結合
        final_result = {
            "diagnosis": full_diagnosis,
            "proposals": proposals_data
        }
        
        print("--- Diagnosis process completed successfully. Sending response. ---")
        return jsonify(final_result)

    except Exception as e:
        # 予期せぬエラーが発生した場合のログ出力とエラーレスポンス
        print(f"An unexpected error occurred during /diagnose: {e}")
        traceback.print_exc()
        # フロントエンドに分かりやすいエラーメッセージを返します。
        error_message = f"サーバー側でエラーが発生しました。しばらくしてからもう一度お試しください。(詳細: {str(e)})"
        return jsonify({"error": error_message}), 500

# --- メイン実行ブロック ---
if __name__ == '__main__':
    # Render.comなどのホスティングサービスはPORT環境変数を参照します
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
