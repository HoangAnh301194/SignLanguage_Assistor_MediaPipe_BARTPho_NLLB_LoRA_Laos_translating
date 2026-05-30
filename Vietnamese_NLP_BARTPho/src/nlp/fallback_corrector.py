TOKEN_MAP = {
    "toi": "tôi",
    "ban": "bạn",
    "can": "cần",
    "muon": "muốn",

    "uong_nuoc": "uống nước",
    "uong": "uống",
    "nuoc": "nước",

    "an_com": "ăn cơm",
    "an": "ăn",
    "com": "cơm",

    "giup_do": "giúp đỡ",
    "giup": "giúp",
    "do": "đỡ",

    "dau": "đau",
    "bung": "bụng",
    "dau_dau": "đau đầu",
    "met": "mệt",

    "benh_vien": "bệnh viện",
    "benh": "bệnh",
    "vien": "viện",

    "nha_ve_sinh": "nhà vệ sinh",
    "nha": "nhà",
    "ve": "vệ",
    "sinh": "sinh",

    "ve_nha": "về nhà",
    "dong_y": "đồng ý",
    "dong": "đồng",
    "y": "ý",
    "khong": "không",

    "nguy_hiem": "nguy hiểm",
    "goi_nguoi_than": "gọi người thân",
    "goi": "gọi",
    "nguoi": "người",
    "than": "thân",

    "xin_chao": "xin chào",
    "cam_on": "cảm ơn",
    "xin_loi": "xin lỗi",

    "bac_si": "bác sĩ",
    "bac": "bác",
    "si": "sĩ",

    "cap_cuu": "cấp cứu",
    "cap": "cấp",
    "cuu": "cứu",

    "nghe_duoc": "nghe được",
    "nghe": "nghe",
    "duoc": "được",

    "hieu": "hiểu",
    "noi": "nói",
    "lai": "lại",
    "gap": "gặp",
    "khoe": "khỏe",
    "nghi": "nghỉ",
    "ngu": "ngủ"
}


def normalize_raw_text(raw: str) -> str:
    tokens = raw.strip().split()
    words = []

    for token in tokens:
        words.append(TOKEN_MAP.get(token, token.replace("_", " ")))

    return " ".join(words)


def to_sentence(text: str) -> str:
    text = text.strip()

    if not text:
        return ""

    text = text[0].upper() + text[1:]

    if text[-1] not in [".", "?", "!"]:
        text += "."

    return text
