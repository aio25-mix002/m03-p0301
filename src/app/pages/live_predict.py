"""
Streamlit page for making live predictions on new abstracts.  The page
leverages the vectoriser and model stored in ``st.session_state`` by the
Experiments page.  Users can enter an abstract and receive the predicted
primary category.  If no model has been trained yet, the page prompts the
user to train one first.
"""

import streamlit as st
import re
import numpy as np

st.title("Live Prediction")
st.write(
    "Nhập tóm tắt bài báo khoa học và nhấn **Dự đoán** để nhận chủ đề dự đoán."
)

if "model" not in st.session_state or "vectorizer" not in st.session_state:
    st.warning(
        "Bạn cần huấn luyện mô hình trước khi dự đoán. "
        "Hãy tới tab **Experiments** để chọn phương pháp và mô hình rồi huấn luyện."
    )
else:
    user_input = st.text_area("Tóm tắt bài báo:", height=200)
    if st.button("Dự đoán"):
        if not user_input.strip():
            st.error("Vui lòng nhập nội dung tóm tắt hợp lệ.")
        else:
            # Basic preprocessing consistent with training pipeline
            cleaned = user_input.strip().replace("\n", " ")
            cleaned = re.sub(r"[^\w\s]", "", cleaned)
            cleaned = re.sub(r"\d+", "", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()

            vectoriser = st.session_state["vectorizer"]
            model = st.session_state["model"]
            vec = vectoriser.transform([cleaned])

            # Handle dense requirement for GaussianNB
            if model.__class__.__name__ == "GaussianNB" and hasattr(vec, "toarray"):
                pred = model.predict(vec.toarray())
            else:
                pred = model.predict(vec)

            pred_label = pred[0]
            st.success(f"Chủ đề dự đoán: **{pred_label}**")

            # Display additional context: model, vectorisation and imbalance handling
            model_name = st.session_state.get("model_name", "N/A")
            vectoriser_name = st.session_state.get("vectoriser_name", "N/A")
            imbalance_name = st.session_state.get("imbalance_option", "N/A")
            st.write(
                f"**Mô hình:** {model_name} | **Vector hóa:** {vectoriser_name} | "
                f"**Xử lý mất cân bằng:** {imbalance_name}"
            )

            # Attempt a simple explanation for linear models with BoW/TF-IDF.
            explanation = None
            try:
                base_vect = vectoriser
                # If the vectoriser is a Pipeline (e.g., TF-IDF + SVD or FeatureUnion),
                # attempt to extract the underlying TF-IDF or CountVectorizer to get feature names.
                if hasattr(vectoriser, "named_steps"):
                    # Pipelines expose steps in named_steps.  Use the first step that has
                    # get_feature_names_out if available.
                    for step_name, step_obj in vectoriser.named_steps.items():
                        if hasattr(step_obj, "get_feature_names_out"):
                            base_vect = step_obj
                            break
                elif hasattr(vectoriser, "transformer_list"):
                    # FeatureUnion stores a list of (name, transformer) tuples
                    for name, transformer in vectoriser.transformer_list:
                        if hasattr(transformer, "get_feature_names_out"):
                            base_vect = transformer
                            break
                if hasattr(base_vect, "get_feature_names_out"):
                    feature_names = base_vect.get_feature_names_out()
                    # Convert sparse vector to dense
                    vec_dense = vec.toarray()[0] if hasattr(vec, "toarray") else np.array(vec)[0]
                    class_idx = list(model.classes_).index(pred_label)
                    if model_name == "Logistic Regression" and hasattr(model, "coef_"):
                        coef = model.coef_[class_idx]
                        contributions = vec_dense * coef
                    elif model_name == "Naive Bayes" and hasattr(model, "feature_log_prob_"):
                        # For Multinomial Naive Bayes, use log probabilities of features
                        coef = model.feature_log_prob_[class_idx]
                        contributions = vec_dense * coef
                    else:
                        contributions = None
                    if contributions is not None:
                        top_indices = np.argsort(contributions)[::-1]
                        explanation = []
                        for idx in top_indices:
                            # Only show top 5 non-zero contributions
                            if len(explanation) >= 5:
                                break
                            if contributions[idx] != 0:
                                explanation.append((feature_names[idx], float(contributions[idx])))
            except Exception:
                explanation = None

            if explanation:
                # Build a colour-coded string highlighting important words.  The more
                # intense the red colour, the higher the contribution.  Only words in
                # the explanation list are coloured; others remain default.
                token_scores = {w.lower(): abs(s) for w, s in explanation}
                max_score = max(token_scores.values()) if token_scores else 0
                tokens = cleaned.split()
                html_tokens: list[str] = []
                for t in tokens:
                    score = token_scores.get(t.lower(), 0.0)
                    if max_score > 0 and score > 0:
                        alpha = score / max_score
                        # Limit alpha to [0.2, 1] for visibility
                        alpha = 0.2 + 0.8 * alpha
                        colour = f"rgba(255, 0, 0, {alpha:.2f})"
                        html_tokens.append(f"<span style='color:{colour}'>{t}</span>")
                    else:
                        html_tokens.append(t)
                highlighted_html = " ".join(html_tokens)
                st.subheader(
                    "Giải thích dự đoán (màu đỏ đậm = từ quan trọng)"
                )
                st.markdown(highlighted_html, unsafe_allow_html=True)
            else:
                st.info(
                    "Không thể giải thích dự đoán cho cấu hình hiện tại. "
                    "Hiện chỉ hỗ trợ Logistic Regression hoặc Naive Bayes với BoW/TF-IDF."
                )
