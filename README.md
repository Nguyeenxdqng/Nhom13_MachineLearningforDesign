# Dự Án Phân Tích - Gợi Ý Phong Cách Thiết Kế
```

```

## DEMO APP

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://app-starter-kit.streamlit.app/)

## GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/app-starter-kit?quickstart=1)

## Các Tính Năng Chính

Dự án này tập trung vào việc xây dựng một hệ thống thông minh hỗ trợ các nhà thiết kế và người dùng tìm kiếm, phân tích và nhận gợi ý về phong cách không gian/sản phẩm dựa trên công nghệ học máy.
- Phân loại Phong cách & Chất liệu: Tự động nhận diện và phân loại phong cách thiết kế (Style) cùng chất liệu (Material) từ mô tả văn bản của người dùng thông qua các mô hình học máy được huấn luyện sẵn ($style\_model.pkl$, $material\_model.pkl$) phối hợp với kỹ thuật trích xuất đặc trưng TF-IDF.
- Tìm kiếm & Gợi ý Thông minh: Sử dụng mô hình nhúng câu SentenceTransformer (all-MiniLM-L6-v2) để tính toán độ tương đồng ngữ nghĩa, từ đó đưa ra các gợi ý thiết kế phù hợp nhất với nhu cầu cụ thể của người dùng.
- Giao diện Trực quan: Tích hợp ứng dụng Web tương tác thông qua thư viện Streamlit, cho phép người dùng nhập yêu cầu và nhận kết quả phân tích cùng bảng màu gợi ý trực tiếp theo thời gian thực.

## Tài liệu Tham khảo

Dưới đây là danh sách các tài liệu, thư viện và tài nguyên hữu ích được sử dụng hoặc có liên quan trực tiếp đến các công nghệ trong dự án này:
- Streamlit Documentation: Hướng dẫn toàn diện về cách xây dựng, tối ưu hóa bộ nhớ cache (@st.cache_resource) và triển khai ứng dụng giao diện web cho các dự án Data Science.
- Scikit-Learn & Joblib: Tài liệu về cách huấn luyện các mô hình phân loại text, xử lý ngôn ngữ tự nhiên với TF-IDF Vectorizer và cách đóng gói/xuất mô hình bằng joblib.
- Sentence Transformers (Hugging Face): Khám phá chi tiết về kiến trúc mô hình all-MiniLM-L6-v2 và cách ứng dụng các vector nhúng (embeddings) vào bài toán tìm kiếm ngữ nghĩa (Semantic Search).
