from flask import Flask, request, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import base64

app = Flask(__name__)

# تحميل النموذج والمرمز
model_name = "Tencent-Hunyuan/Hy-MT2-1.8B"  # اختر الحجم المناسب
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.route('/subtitle', methods=['POST'])
def get_subtitle():
    data = request.json
    video_id = data.get('video_id')
    language = data.get('language', 'en')  # اللغة الافتراضية هي الإنجليزية

    # افترض أن لدينا نص الفيديو هنا (يمكنك استخراج النص من الفيديو باستخدام مكتبات مثل moviepy)
    video_text = "Example video text to be translated."

    # الترميز والترجمة باستخدام Hy-MT2
    inputs = tokenizer(video_text, return_tensors="pt", max_length=512, truncation=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    # إنشاء SRT
    srt_content = f"1\n00:00:00,000 --> 00:00:05,000\n{translated_text}"

    return jsonify({
        "id": f"{video_id}_{language}",
        "title": f"{language.upper()} Subtitles",
        "language": language,
        "downloadUrl": f"data:text/plain;charset=utf-8;base64,{base64.b64encode(srt_content.encode('utf-8')).decode('utf-8')}",
        "types": ["subtitle"]
    })

if __name__ == '__main__':
    app.run(port=5000)
