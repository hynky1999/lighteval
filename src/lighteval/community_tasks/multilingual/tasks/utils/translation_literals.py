from typing import Literal


# This is experimental flag if activated languages without sentence space separation will be evaluated that way
LANGS = Literal["ar", "en", "fr", "hi", "ru", "sw", "te", "th", "tr", "zh", "eu", "jp", "id", "de", "it", "vi"]

LANG_NAMES = {
    "arabic": "ar",
    "bengali": "bn",
    "english": "en",
    "finnish": "fi",
    "french": "fr",
    "indonesian": "id",
    "korean": "ko",
    "russian": "ru",
    "swahili": "sw",
    "telugu": "te",
    "afrikaans": "af",
    "chinese": "zh",
    "italian": "it",
    "javanese": "jv",
    "hindi": "hi",
    "turkish": "tr",
    "portuguese": "pt",
    "thai": "th",
    "vietnamese": "vi",
    "japanese": "jp",
    "basque": "eu",
    "german": "de",
}

NO_PUNCT_LANGS = ["th"]

LANG_NAMES_INVERTED = {v: k for k, v in LANG_NAMES.items()}


QUESTION = {
    "ar": "سؤال",
    "en": "Question",
    "fr": "Question",
    "hi": "सवाल",
    "ru": "Вопрос",
    "sw": "Swali",
    "te": "ప్రశ్న",
    "th": "คำถาม",
    "tr": "Soru",
    "eu": "Galdera",
    "zh": "问题",
    "jp": "質問",
    "id": "pertanyaan",
    "de": "frage",
    "it": "domanda",
    "vi": "câu hỏi",
}

OPTIONS = {
    "th": "ตัวเลือก",
    "tr": "Seçenekler",
    "fr": "Possibilités",
    "ar": "خيارات",
    "zh": "选项",
    "it": "opzioni",
}

ANSWER = {
    "ar": "إجابة",
    "en": "Answer",
    "fr": "Réponse",
    "hi": "उत्तर",
    "ru": "ответ",
    "sw": "Jibu",
    "te": "జవాబు",
    "th": "คำตอบ",
    "tr": "Cevap",
    "eu": "Erantzuna",
    "zh": "答案",
    "jp": "回答",
    "id": "jawaban",
    "de": "antwort",
    "it": "risposta",
    "vi": "trả lời",
}


# Harness cfg
# We sure the XNLI definitions from eval harness
NLI_QUESTION = {
    "ar": "صحيح",
    "en": "right",
    "fr": "n'est-ce pas",
    "hi": "है ना",
    "ru": "не так ли",
    "sw": "sahihi",
    "te": "కదా",
    "th": "ใช่ไหม",
    "tr": "değil mi",
    "zh": "是不是",
    "eu": "ezta",
    "jp": "でしょうか",
    "id": "kan",
    "de": "richtig",
    "it": "vero",
    "vi": "đúng",
}

ENTAILMENT_LABELS = {
    "ar": "نعم",
    "en": "Yes",
    "fr": "Oui",
    "hi": "हाँ",
    "ru": "Да",
    "sw": "Ndiyo",
    "te": "అవును",
    "th": "ใช่",
    "tr": "Evet",
    "zh": "是的",
    "eu": "Bai",
    "jp": "はい",
    "id": "ya",
    "de": "ja",
    "it": "sì",
    "vi": "có",
}

NEUTRAL_LABELS = {
    "ar": "كذلك",
    "en": "Also",
    "fr": "De plus",
    "hi": "साथ ही",
    "ru": "К тому же",
    "sw": "Pia",
    "th": "และ",
    "tr": "Ayrıca",
    "te": "అలాగే",
    "zh": "而且",
    "eu": "Halaber",
    "jp": "また",
    "id": "juga",
    "de": "auch",
    "it": "inoltre",
    "vi": "cũng",
}

CONTRADICTION_LABELS = {
    "ar": "لا",
    "en": "No",
    "fr": "Non",
    "hi": "नहीं",
    "ru": "Нет",
    "sw": "Hapana",
    "te": "కాదు",
    "th": "ไม่",
    "tr": "Hayır",
    "zh": "不是",
    "eu": "Ezta",
    "jp": "いいえ",
    "id": "Tidak",
    "de": "Nein",
    "it": "No",
    "vi": "Không",
}

IMPOSSIBLE = {
    "fr": "Impossible",
}


CORRECT_LABELS = {
    "ar": "صح",
    "jp": "正解",
}

INCORRECT_LABELS = {
    "ar": "خطأ",
    "jp": "不正解",
}

YES_LABELS = {
    "ar": "نعم",
    "hi": "हाँ",
    "fr": "Oui",
    "jp": "はい",
}

NO_LABELS = {
    "ar": "لا",
    "hi": "नहीं",
    "fr": "Non",
    "jp": "いいえ",
}

CAUSE_LABELS = {
    "ar": "لأن",
    "en": "because",
    "hi": "क्योंकि",
    "sw": "kwa sababu",
    "zh": "因为",
    "ta": "காரணமாக",
    "te": "ఎందుకంటే",
    "th": "เพราะ",
    "tr": "çünkü",
    "ru": "потому что",
    "eu": "izan ere",
    "jp": "なぜなら",
    "id": "karena",
    "de": "weil",
    "it": "perchè",
    "vi": "vì",
}

EFFECT_LABELS = {
    "ar": "لذلك",
    "en": "therefore",
    "hi": "इसलिए",
    "sw": "kwa hiyo",
    "zh": "所以",
    "ta": "எனவே",
    "te": "అందువలన",
    "th": "ดังนั้น",
    "tr": "bu yüzden",
    "ru": "поэтому",
    "eu": "beraz",
    "jp": "なので",
    "id": "maka",
    "de": "deshalb",
    "it": "quindi",
    "vi": "do đó",
}


# Thai doesn't ahve full stop instead they use spac
FULL_STOP = {
    "ar": ".",
    "en": ".",
    "hi": "।",
    "fr": ".",
    "sw": ".",
    "th": "",
    "zh": "。",
    "te": ".",
    "tr": ".",
    "ru": ".",
    "eu": ".",
    "jp": "。",
    "id": ".",
    "de": ".",
    "it": ".",
    "vi": ".",
}

WORD_SPACE = {
    "ar": " ",
    "en": " ",
    "hi": " ",
    "fr": " ",
    "ru": " ",
    "sw": " ",
    "te": " ",
    "th": "",
    "tr": " ",
    "zh": "",
    "eu": " ",
    "jp": "",
    "id": " ",
    "de": " ",
    "it": " ",
    "vi": " ",
}

SENTENCE_SPACE = {
    "ar": " ",
    "en": " ",
    "hi": " ",
    "fr": " ",
    "ru": " ",
    "sw": " ",
    "te": " ",
    "th": " ",
    "tr": " ",
    "zh": "",
    "eu": " ",
    "jp": "",
    "id": " ",
    "de": " ",
    "it": " ",
    "vi": " ",
}

COMMA = {
    "ar": "،",
    "en": ",",
    "hi": ",",
    "fr": ",",
    "ru": ",",
    "sw": ",",
    "te": ",",
    "th": ",",
    "tr": ",",
    "zh": "，",
    "eu": ",",
    "jp": "、",
    "id": ",",
    "de": ",",
    "it": ",",
    "vi": ",",
}


QUESTION_MARK = {
    "ar": "؟",
    "en": "?",
    "hi": "?",
    "fr": "?",
    "ru": "?",
    "sw": "?",
    "te": "?",
    "th": "?",
    "tr": "?",
    "zh": "？",
    "eu": "?",
    "jp": "？",
    "id": "?",
    "de": "?",
    "it": "?",
    "vi": "?",
}

COLON = {
    "ar": ":",
    "en": ":",
    "hi": ":",
    "fr": ":",
    "ru": ":",
    "sw": ":",
    "te": ":",
    "th": ":",
    "tr": ":",
    "zh": "：",
    "eu": ":",
    "jp": "：",
    "id": ":",
    "de": ":",
    "it": ":",
    "vi": ":",
}


AND = {
    "zh": "和",
}

OR = {
    "zh": "或",
    "jp": "または",
    "id": "atau",
    "de": "oder",
    "it": "o",
    "vi": "hoặc",
    "eu": "ala",
}