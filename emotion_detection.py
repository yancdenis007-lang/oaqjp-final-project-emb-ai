"""Emotion Detection module using Watson NLP library."""
import json
import requests


def emotion_detector(text_to_analyze):
    """Analyzes text for emotions using Watson NLP API and returns

    a dictionary of emotion scores and the dominant emotion.
    """
    # 1. Проверка на пустой ввод (защита от ошибок)
    if not text_to_analyze or not text_to_analyze.strip():
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    url = (
        "https://sn-watson-emotion.labs.skills.network/v1/"
        "watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )
    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }
    input_json = {"raw_document": {"text": text_to_analyze}}

    # 2. Отправка POST-запроса к API
    response = requests.post(url, json=input_json, headers=headers, timeout=10)

    # 3. Обработка ошибки со статусом 400 (пустой или неверный запрос)
    if response.status_code == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    # 4. Превращаем текст ответа в словарь Python
    formatted_response = json.loads(response.text)
    emotions = formatted_response["emotionPredictions"][0]["emotion"]

    anger_score = emotions["anger"]
    disgust_score = emotions["disgust"]
    fear_score = emotions["fear"]
    joy_score = emotions["joy"]
    sadness_score = emotions["sadness"]

    # 5. Находим доминирующую эмоцию (у которой самый высокий балл)
    emotion_scores = {
        "anger": anger_score,
        "disgust": disgust_score,
        "fear": fear_score,
        "joy": joy_score,
        "sadness": sadness_score,
    }
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)

    # 6. Возвращаем итоговый словарь
    return {
        "anger": anger_score,
        "disgust": disgust_score,
        "fear": fear_score,
        "joy": joy_score,
        "sadness": sadness_score,
        "dominant_emotion": dominant_emotion,
    }