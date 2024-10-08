from typing import Literal


# This is experimental flag if activated languages without sentence space separation will be evaluated that way
LANGS = Literal["ar", "en", "fr", "hi", "ru", "sw", "te", "th", "tr", "zh", "eu"]

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
}

OPTIONS = {
    "th": "ตัวเลือก",
    "tr": "Seçenekler",
    "fr": "Possibilités",
    "ar": "خيارات",
    "zh": "选项"
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
    "eu": "Ez",
}

IMPOSSIBLE = {
    "fr": "Impossible",
}


CORRECT_LABELS = {
    "ar": "صح"
}

INCORRECT_LABELS = {
    "ar": "خطأ"
}

YES_LABELS = {
    "ar": "نعم",
    "hi": "हाँ",
    "fr": "Oui"
}

NO_LABELS = {
    "ar": "لا",
    "hi": "नहीं",
    "fr": "Non"
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
    "eu": "zaren",
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
    "eu": "horregatik",
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
}


AND = {
    "zh": "和",
}

OR = {
    "zh": "或",
}