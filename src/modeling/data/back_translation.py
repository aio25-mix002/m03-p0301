# Trên Google Colab: cài đặt thư viện cần thiết
#!pip install -q datasets googletrans==4.0.0rc1 scikit-learn

import pandas as pd
from datasets import load_dataset
from googletrans import Translator
from sklearn.model_selection import train_test_split
import collections
from src.configuration.configuration_manager import ConfigurationManager
SETTINGS = ConfigurationManager.load()
# --- Bước 1: Tải và chuẩn bị dữ liệu gốc ---
# Tải dữ liệu từ Hugging Face
raw = load_dataset("UniverseTBD/arxiv-abstracts-large", split="train", cache_dir=SETTINGS.data.external_huggingface_dir)

# Lọc lấy 5 lớp chính và các bản ghi đơn nhãn
def filter_single_label(ex):
    cats = ex["categories"].split()
    return len(cats) == 1 and cats[0].split(".")[0] in ["astro-ph", "cond-mat", "cs", "math", "physics"]

filtered = raw.filter(filter_single_label)

# Lấy 1000 mẫu để làm ví dụ
samples = filtered.shuffle(seed=42).select(range(1000))
texts = [s["abstract"] for s in samples]
labels = [c.split(".")[0] for c in samples["categories"]]

# --- Bước 2: Chia dữ liệu thành tập Train và Test TRƯỚC KHI tăng cường ---
# Đây là bước quan trọng nhất để tránh rò rỉ dữ liệu.
# Tập test (X_test, y_test) sẽ được giữ nguyên và không bị thay đổi.
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Kích thước tập train ban đầu: {len(X_train)} mẫu")
print(f"Kích thước tập test: {len(X_test)} mẫu")
print("-" * 30)


# --- Bước 3: Áp dụng Back Translation CHỈ trên tập Train ---
translator = Translator()

def back_translate(text, inter="vi"):
    """Dịch văn bản sang ngôn ngữ trung gian rồi dịch ngược lại tiếng Anh."""
    try:
        tmp = translator.translate(text, dest=inter).text
        return translator.translate(tmp, dest="en").text
    except Exception as e:
        print(f"Lỗi dịch, sử dụng lại văn bản gốc. Lỗi: {e}")
        return text

# Xác định lớp đa số trong tập train để cân bằng
counter = collections.Counter(y_train)
max_count = max(counter.values()) if counter else 0

print("Phân bố nhãn trong tập train TRƯỚC khi tăng cường:")
print(counter)
print("-" * 30)

aug_texts, aug_labels = [], []
# Lặp qua từng lớp trong tập train để tăng cường dữ liệu
for cat in counter:
    # Lấy tất cả văn bản của lớp đó trong tập train
    cat_texts = [t for t, lb in zip(X_train, y_train) if lb == cat]
    n_current = len(cat_texts)

    if n_current > 0 and n_current < max_count:
        needed = max_count - n_current  # Số lượng mẫu cần bổ sung
        for i in range(needed):
            # Lặp lại các mẫu gốc của lớp đó để dịch
            original_text = cat_texts[i % n_current]
            translated_text = back_translate(original_text)

            aug_texts.append(translated_text)
            aug_labels.append(cat)

# --- Bước 4: Gộp dữ liệu train gốc và dữ liệu đã tăng cường ---
# Tập train cuối cùng sẽ bao gồm cả dữ liệu gốc và dữ liệu mới
final_X_train = X_train + aug_texts
final_y_train = y_train + aug_labels

print(f"Kích thước tập train SAU khi tăng cường: {len(final_X_train)} mẫu")
print("Phân bố nhãn trong tập train SAU khi tăng cường:")
print(collections.Counter(final_y_train))
print("-" * 30)


# --- Bước 5: Lưu kết quả ---
# Bây giờ bạn có thể lưu các tập dữ liệu này để sử dụng cho việc huấn luyện và đánh giá
df_train = pd.DataFrame({"text": final_X_train, "label": final_y_train})
df_test = pd.DataFrame({"text": X_test, "label": y_test})

# Lưu ra file CSV
df_train.to_csv("arxiv_train_augmented.csv", index=False)
df_test.to_csv("arxiv_test_untouched.csv", index=False)

print("Đã lưu tập train đã tăng cường vào file: arxiv_train_augmented.csv")
print("Đã lưu tập test còn nguyên vẹn vào file: arxiv_test_untouched.csv")
