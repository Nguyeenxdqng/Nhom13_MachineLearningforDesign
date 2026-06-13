# 🎨 Dự Án Phân Tích - Gợi Ý Phong Cách Thiết Kế (Interior Design Recommendation System)


📌 Giới Thiệu Dự Án
Trong lĩnh vực thiết kế nội thất, việc nắm bắt chính xác ý tưởng của khách hàng qua mô tả bằng lời nói hoặc văn bản là một thách thức lớn. Dự án này được phát triển nhằm xây dựng một **Hệ thống trợ lý thông minh**, áp dụng các mô hình Học máy (Machine Learning) và Xử lý ngôn ngữ tự nhiên (NLP) để tự động hóa quy trình phân tích phong cách và đưa ra các gợi ý thiết kế không gian sống tối ưu.

### 🎯 Mục tiêu chính:
* **Hỗ trợ nhà thiết kế:** Tiết kiệm thời gian phân tích yêu cầu sơ bộ từ khách hàng, định hình nhanh phong cách chủ đạo.
* **Tối ưu trải nghiệm người dùng:** Giúp người không có chuyên môn dễ dàng tìm kiếm cảm hứng và nhận về các đề xuất trực quan (chất liệu, bảng màu, ý tưởng không gian) dựa trên mong muốn cá nhân.

### ⚙️ Quy trình xử lý (Pipeline):
1. **Dữ liệu đầu vào:** Người dùng nhập một đoạn mô tả ngắn về không gian mong muốn (ví dụ: *"Phòng khách nhỏ gọn, ấm cúng với nội thất gỗ tối giản"*).
2. **Trích xuất & Phân loại:** Hệ thống sử dụng TF-IDF kết hợp mô hình phân loại để xác định chính xác **Style (Phong cách)** và **Material (Chất liệu chính)**.
3. **Tìm kiếm ngữ nghĩa (Semantic Search):** Sử dụng mô hình Vector hóa câu để tính toán độ tương đồng, từ đó truy xuất ra các mẫu thiết kế phù hợp nhất trong cơ sở dữ liệu.
4. **Phản hồi trực quan:** Xuất kết quả phân tích, gợi ý bảng màu phối hợp và các mẫu thiết kế liên quan theo thời gian thực trên giao diện Web.

## Các Tính Năng Chính

Dự án này tập trung vào việc xây dựng một hệ thống thông minh hỗ trợ các nhà thiết kế và người dùng tìm kiếm, phân tích và nhận gợi ý về phong cách không gian/sản phẩm dựa trên công nghệ học máy.
- Phân loại Phong cách & Chất liệu: Tự động nhận diện và phân loại phong cách thiết kế (Style) cùng chất liệu (Material) từ mô tả văn bản của người dùng thông qua các mô hình học máy được huấn luyện sẵn phối hợp với kỹ thuật trích xuất đặc trưng TF-IDF.
- Tìm kiếm & Gợi ý Thông minh: Sử dụng mô hình nhúng câu SentenceTransformer (all-MiniLM-L6-v2) để tính toán độ tương đồng ngữ nghĩa, từ đó đưa ra các gợi ý thiết kế phù hợp nhất với nhu cầu cụ thể của người dùng.
- Giao diện Trực quan: Tích hợp ứng dụng Web tương tác thông qua thư viện Streamlit, cho phép người dùng nhập yêu cầu và nhận kết quả phân tích cùng bảng màu gợi ý trực tiếp theo thời gian thực.

## Tài liệu Tham khảo

Dưới đây là danh sách các tài liệu, thư viện và tài nguyên hữu ích được sử dụng hoặc có liên quan trực tiếp đến các công nghệ trong dự án này:
- Streamlit Documentation: Hướng dẫn toàn diện về cách xây dựng, tối ưu hóa bộ nhớ cache (@st.cache_resource) và triển khai ứng dụng giao diện web cho các dự án Data Science.
- Scikit-Learn & Joblib: Tài liệu về cách huấn luyện các mô hình phân loại text, xử lý ngôn ngữ tự nhiên với TF-IDF Vectorizer và cách đóng gói/xuất mô hình bằng joblib.
- Sentence Transformers (Hugging Face): Khám phá chi tiết về kiến trúc mô hình all-MiniLM-L6-v2 và cách ứng dụng các vector nhúng (embeddings) vào bài toán tìm kiếm ngữ nghĩa (Semantic Search).

## DEMO APP

[![Streamlit App](https://nhom13machinelearningfordesign-inspirationrecommendationsystem.streamlit.app/)]
