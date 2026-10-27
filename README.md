# 🍄 Mushroom Classification — Khai phá luật kết hợp (Apriori vs FP-Growth)

Đồ án khai phá luật kết hợp trên tập dữ liệu **Mushroom** (UCI / Kaggle): tìm các tổ hợp đặc điểm hình thái của nấm thường đi cùng nhãn **ăn được (edible)** hoặc **độc (poisonous)**, đồng thời so sánh hai thuật toán **Apriori** và **FP-Growth**.

> ⚠️ **Cảnh báo an toàn:** kết quả chỉ phục vụ học tập và nghiên cứu. **Không dùng các luật trong dự án này để quyết định có ăn nấm ngoài tự nhiên hay không.** Nhiều loài nấm độc trông gần như giống hệt nấm ăn được; ngộ độc nấm có thể gây tử vong.

---

## 1. Giới thiệu bài toán

### Bối cảnh
Phân biệt nấm ăn được và nấm độc chỉ bằng quan sát bên ngoài là việc khó và rủi ro. Tập dữ liệu Mushroom mô tả nấm bằng các đặc điểm hình thái (mũ, phiến mang, cuống, vòng, mùi, môi trường sống…), rất phù hợp để khai phá các quy luật dạng *“nếu có đặc điểm A và B thì thường là loại C”*.

### Mục tiêu
- Tiền xử lý và chuẩn hóa dữ liệu nấm.
- Biểu diễn mỗi mẫu nấm thành một **transaction** (tập các cặp `thuộc_tính=giá_trị`).
- Khai phá tập phổ biến và luật kết hợp bằng **Apriori** và **FP-Growth**.
- So sánh hai thuật toán (số tập phổ biến, số luật, thời gian) và chọn ra các luật tiêu biểu liên quan đến nhãn `class`.

### Câu hỏi nghiên cứu
1. Những đặc điểm nào đứng đầu về mức độ liên hệ với nhãn ăn được / độc?
2. Những tổ hợp đặc điểm nào xuất hiện cùng nhau thường xuyên (tập phổ biến)?
3. Luật nào có độ tin cậy (confidence) và độ nâng (lift) cao khi dự đoán `edible` hoặc `poisonous`?
4. Apriori và FP-Growth có cho cùng kết quả không, và khác nhau thế nào về thời gian chạy?

### Phạm vi
- Dữ liệu: một tập duy nhất, toàn bộ thuộc tính dạng phân loại (categorical).
- Kỹ thuật: khai phá luật kết hợp (Apriori, FP-Growth) với `minsup ∈ {0.3, 0.4, 0.5}`, `minconf ∈ {0.6, 0.7, 0.8}`.
- Không xây dựng mô hình phân loại dự đoán trên dữ liệu mới.

### Ý nghĩa của kết quả
Các luật cho thấy đặc điểm nào của nấm gắn chặt với nhãn độc / ăn được trong tập dữ liệu này, giúp minh họa cách khai phá luật kết hợp trên dữ liệu phân loại và so sánh hiệu năng hai thuật toán.

### Giới hạn và cảnh báo an toàn
- Luật chỉ phản ánh **tập dữ liệu này** (8.124 mẫu, một số loài trong họ Agaricus và Lepiota), không đại diện cho mọi loài nấm.
- Luật có confidence cao vẫn **không phải bằng chứng chắc chắn** về an toàn khi ăn.
- Xem thêm mục 8.

---

## 2. Dữ liệu

| Mục | Nội dung |
|---|---|
| Tên dataset | **Mushroom Classification** |
| Nguồn | Kaggle (bản đăng lại), gốc từ **UCI Machine Learning Repository – Mushroom Data Set** |
| Số mẫu | **8.124** |
| Số thuộc tính | 23 cột ở dữ liệu thô (1 nhãn + 22 đặc trưng) → **22 cột** sau làm sạch (1 nhãn + **21 đặc trưng**) |
| Nhãn (`class`) | `e` = **edible** (ăn được): 4.208 mẫu (51,8%) · `p` = **poisonous** (độc): 3.916 mẫu (48,2%) |
| Kiểu dữ liệu | Toàn bộ là categorical, mỗi giá trị là một ký tự |

### Cách biểu diễn transaction
Mỗi dòng (một cây nấm) được chuyển thành một transaction gồm các item dạng `tên_cột=giá_trị`, ví dụ `odor=n`, `gill-size=b`, `class=e`. Sau đó transaction được mã hóa one-hot thành ma trận **8.124 × 118** (118 item khác nhau) để đưa vào Apriori / FP-Growth.

### Xử lý giá trị thiếu `?` ở `stalk-root`
- Cột `stalk-root` có **2.480 giá trị `?`** (≈ 30,5%). Các cột còn lại không thiếu giá trị.
- **Quyết định:** giữ `?` như một danh mục độc lập (*missing – thiếu dữ liệu*), không xóa dòng (tránh mất ~1/3 dữ liệu) và không điền bằng mode (tránh làm sai lệch đặc trưng thực tế).

### Các bước làm sạch khác
- Không có dòng trùng lặp; không có giá trị ngoài chuẩn UCI.
- Loại bỏ cột **`veil-type`** vì chỉ có một giá trị duy nhất (không mang thông tin phân biệt).

### Giới hạn của dữ liệu
- Chỉ gồm các mẫu mô phỏng từ 23 loài nấm mang (gilled mushrooms) → khó khái quát cho nấm nói chung.
- Mùi (`odor`) rất mạnh trong dữ liệu này nhưng là đặc điểm chủ quan khi quan sát ngoài đời thực.
- Có 30,5% thiếu `stalk-root`; một số mã trong dictionary không xuất hiện trong dữ liệu (ví dụ `gill-attachment`: `d`, `n`; `ring-type`: `c`, `s`, `z`; `stalk-root`: `u`, `z`; `gill-spacing`: `d`).

---

## 3. Data Dictionary

Bảng đầy đủ nằm ở [`docs/data_dictionary.xlsx`](docs/data_dictionary.xlsx); tóm tắt như sau:

| # | Tên cột | Ý nghĩa | Vai trò | Giá trị (ký hiệu = nghĩa) |
|---|---|---|---|---|
| 1 | `class` | Loại nấm | Nhãn đích | e = edible (ăn được); p = poisonous (độc) |
| 2 | `cap-shape` | Hình dạng mũ | Đặc trưng | b = bell (hình chuông); c = conical (hình nón); x = convex (lồi); f = flat (phẳng); k = knobbed (có núm); s = sunken (lõm) |
| 3 | `cap-surface` | Bề mặt mũ | Đặc trưng | f = fibrous (xơ); g = grooves (có rãnh); y = scaly (có vảy); s = smooth (nhẵn) |
| 4 | `cap-color` | Màu mũ nấm | Đặc trưng | n = brown (nâu); b = buff (vàng nâu nhạt); c = cinnamon (quế); g = gray (xám); r = green (xanh lá); p = pink (hồng); u = purple (tím); e = red (đỏ); w = white (trắng); y = yellow (vàng) |
| 5 | `bruises` | Tình trạng bầm tím | Đặc trưng | t = bruises (có bầm); f = no (không bầm) |
| 6 | `odor` | Mùi của nấm | Đặc trưng | a = almond (hạnh nhân); l = anise (hồi); c = creosote (creosote); y = fishy (tanh); f = foul (hôi); m = musty (ẩm mốc); n = none (không mùi); p = pungent (hăng); s = spicy (cay) |
| 7 | `gill-attachment` | Kiểu gắn của phiến mang | Đặc trưng | a = attached (gắn); d = descending (xuống); f = free (tự do); n = notched (khía) |
| 8 | `gill-spacing` | Khoảng cách giữa các phiến mang | Đặc trưng | c = close (sát); w = crowded (dày); d = distant (thưa) |
| 9 | `gill-size` | Kích thước phiến mang | Đặc trưng | b = broad (rộng); n = narrow (hẹp) |
| 10 | `gill-color` | Màu phiến mang | Đặc trưng | k = black (đen); n = brown (nâu); b = buff (vàng nâu nhạt); h = chocolate (sô-cô-la); g = gray (xám); r = green (xanh lá); o = orange (cam); p = pink (hồng); u = purple (tím); e = red (đỏ); w = white (trắng); y = yellow (vàng) |
| 11 | `stalk-shape` | Đặc điểm phần gốc cuống | Đặc trưng | e = enlarging (phình ra); t = tapering (thon dần) |
| 12 | `stalk-root` | Rễ thân | Đặc trưng | b = bulbous (phình củ); c = club (hình chùy); u = cup (hình cốc); e = equal (đều); z = rhizomorphs (thân rễ); r = rooted (có rễ); ? = missing (thiếu dữ liệu) |
| 13 | `stalk-surface-above-ring` | Bề mặt cuống phía trên vòng | Đặc trưng | f = fibrous (xơ); y = scaly (có vảy); k = silky (mượt như lụa); s = smooth (nhẵn) |
| 14 | `stalk-surface-below-ring` | Bề mặt cuống phía dưới vòng | Đặc trưng | f = fibrous (xơ); y = scaly (có vảy); k = silky (mượt như lụa); s = smooth (nhẵn) |
| 15 | `stalk-color-above-ring` | Màu cuống phía trên vòng | Đặc trưng | n = brown (nâu); b = buff (vàng nâu nhạt); c = cinnamon (quế); g = gray (xám); o = orange (cam); p = pink (hồng); e = red (đỏ); w = white (trắng); y = yellow (vàng) |
| 16 | `stalk-color-below-ring` | Màu cuống phía dưới vòng | Đặc trưng | n = brown (nâu); b = buff (vàng nâu nhạt); c = cinnamon (quế); g = gray (xám); o = orange (cam); p = pink (hồng); e = red (đỏ); w = white (trắng); y = yellow (vàng) |
| 17 | `veil-type` *(đã loại bỏ khỏi dữ liệu sạch)* | Loại màn che | Đặc trưng | p = partial (một phần); u = universal (toàn phần) |
| 18 | `veil-color` | Màu màn che | Đặc trưng | n = brown (nâu); o = orange (cam); w = white (trắng); y = yellow (vàng) |
| 19 | `ring-number` | Số vòng | Đặc trưng | n = none (không có); o = one (một); t = two (hai) |
| 20 | `ring-type` | Kiểu vòng | Đặc trưng | c = cobwebby (dạng mạng nhện); e = evanescent (thoáng biến mất); f = flaring (loe); l = large (lớn); n = none (không có); p = pendant (rủ); s = sheathing (bao bọc); z = zone (dạng vòng vùng) |
| 21 | `spore-print-color` | Màu bản in bào tử | Đặc trưng | k = black (đen); n = brown (nâu); b = buff (vàng nâu nhạt); h = chocolate (sô-cô-la); r = green (xanh lá); o = orange (cam); u = purple (tím); w = white (trắng); y = yellow (vàng) |
| 22 | `population` | Đặc điểm/quy mô quần thể | Đặc trưng | a = abundant (phong phú); c = clustered (tụ cụm); n = numerous (nhiều); s = scattered (rải rác); v = several (một số); y = solitary (đơn độc) |
| 23 | `habitat` | Môi trường sống | Đặc trưng | g = grasses (đồng cỏ); l = leaves (lá); m = meadows (đồng cỏ); p = paths (lối đi); u = urban (khu đô thị); w = waste (khu vực chất thải); d = woods (rừng) |

---

## 4. Dữ liệu đã chuẩn hóa (ký hiệu → chữ dễ hiểu)

Dữ liệu sạch gốc dùng ký hiệu một chữ cái (`e`, `p`, `x`, `n`…) rất khó đọc. Dữ liệu được chuẩn hóa **đúng nguyên văn Data Dictionary**: mỗi ký hiệu được thay bằng đúng nhãn ghi trong dictionary (giữ nguyên chữ thường, không tự dịch hay sửa chữ).

| Tệp | Mô tả |
|---|---|
| [`data/processed/Mushroom_Normalized_Dataset.xlsx`](data/processed/Mushroom_Normalized_Dataset.xlsx) | Gồm 4 sheet: **Dữ liệu (theo Dictionary)** — tên cột là cột *Ý nghĩa*, giá trị là nhãn đầy đủ dạng `english (tiếng việt)`; **Dữ liệu (English)** — giữ tên cột gốc, giá trị là phần tiếng Anh; **Bảng ánh xạ** — từng ký hiệu → English / Tiếng Việt / nhãn đầy đủ, kèm số mẫu và tỷ lệ; **Tra cứu đặc trưng (item)** — 118 item trong transaction (`odor=n`) → tên dễ đọc, dùng khi demo để biết một đặc trưng là gì |
| [`data/processed/Mushroom_Normalized_Dataset.csv`](data/processed/Mushroom_Normalized_Dataset.csv) | Bản CSV (UTF-8), tên cột gốc, giá trị tiếng Anh — tiện đưa vào code |
| [`docs/value_mapping.csv`](docs/value_mapping.csv) | Bảng ánh xạ ở dạng CSV |
| [`data/transactions/transactions_normalized.csv`](data/transactions/transactions_normalized.csv) | Ma trận transaction one-hot (8.124 × 118) với tên item dễ đọc, ví dụ `Màu mũ nấm = brown (nâu)` thay cho `cap-color=n` |
| [`results/rules/Mushroom_Rules_Normalized.xlsx`](results/rules/Mushroom_Rules_Normalized.xlsx) | Luật kết hợp viết bằng chữ: sheet **Luật tiêu biểu** (16 luật) và sheet **Luật vế phải có class** (3.083 luật), kèm cột luật gốc để đối chiếu |

**Ví dụ (đúng theo dictionary):**

| Trước (`Cleaned`) | Sau (chuẩn hóa) |
|---|---|
| `class = p` | Loại nấm = **poisonous (độc)** |
| `odor = a` | Mùi của nấm = **almond (hạnh nhân)** |
| `cap-color = n` | Màu mũ nấm = **brown (nâu)** |
| `cap-color = b` | Màu mũ nấm = **buff (vàng nâu nhạt)** |
| `stalk-root = ?` | Rễ thân = **missing (thiếu dữ liệu)** |

**Lưu ý khi chuẩn hóa:**
- Số dòng giữ nguyên **8.124**; phép ánh xạ **đảo ngược được 100%** về ký hiệu gốc; mọi nhãn đều được kiểm tra khớp nguyên văn với ô *Giá trị* trong dictionary.
- Trong cột `habitat`, dictionary ghi cùng nghĩa tiếng Việt “đồng cỏ” cho hai mã khác nhau (`g` = grasses, `m` = meadows). Vì vậy dữ liệu dùng **nhãn đầy đủ `english (tiếng việt)`** — `grasses (đồng cỏ)` và `meadows (đồng cỏ)` — để hai nhóm không bị gộp nhầm.
- Cột `veil-type` có trong dictionary nhưng đã bị loại ở bước làm sạch nên không xuất hiện trong dữ liệu chuẩn hóa.

---

## 5. Cấu trúc thư mục

```
Mushroom-Classification/
├── Mushroom.ipynb                     # Tiền xử lý, EDA, tạo transaction
├── data/
│   ├── raw/mushrooms.csv              # Dữ liệu thô (UCI/Kaggle)
│   ├── processed/
│   │   ├── Mushroom_Preprocessed_Dataset.xlsx
│   │   ├── Mushroom_Preprocessed_Dataset_Cleaned.xlsx   # Dữ liệu sạch (ký hiệu)
│   │   ├── Mushroom_Normalized_Dataset.xlsx             # Dữ liệu chuẩn hóa (chữ)
│   │   └── Mushroom_Normalized_Dataset.csv
│   └── transactions/                  # transactions.csv / .xls / .pkl (ký hiệu) + transactions_normalized.csv (chữ)
├── docs/
│   ├── data_dictionary.xlsx
│   └── value_mapping.csv
├── notebooks/
│   ├── Aprioriipynb.ipynb             # Apriori
│   ├── FP_GROWTH.ipynb                # FP-Growth
│   └── ss_2_THUATTOAN.ipynb           # So sánh Apriori vs FP-Growth
└── results/
    ├── EDA/Mushroom_EDA_Report.xlsx
    ├── comparison/                    # Bảng so sánh hai thuật toán
    ├── figures/so_sanh_Apriori_FPGrowth.png
    └── rules/
        ├── Mushroom_Rules_Optimal.xlsx
        └── Mushroom_Rules_Normalized.xlsx   # Luật viết bằng chữ
```

---

## 6. Quy trình thực hiện

1. **Tiền xử lý** (`Mushroom.ipynb`): đọc `mushrooms.csv`, kiểm tra thiếu / trùng / bất thường, loại `veil-type`.
2. **EDA:** phân phối nhãn, kiểm định Chi-square giữa từng thuộc tính và `class`, phân tích `odor`.
3. **Chuẩn hóa nhãn:** thay ký hiệu bằng chữ theo Data Dictionary (mục 4).
4. **Tạo transaction & one-hot:** ma trận 8.124 × 118.
5. **Khai phá:** Apriori và FP-Growth (`mlxtend`) với `minsup` 0,3 / 0,4 / 0,5 và `minconf` 0,6 / 0,7 / 0,8.
6. **Lọc luật:** bỏ luật có lift ≈ 1 (0,9–1,1), bỏ luật dư thừa và trùng tập, chọn luật tiêu biểu.

### Cài đặt & chạy

```bash
pip install pandas numpy scipy matplotlib mlxtend openpyxl xlrd
jupyter notebook
```

Chạy lần lượt `Mushroom.ipynb` → các notebook trong `notebooks/`. Notebook đọc sheet `3. Dữ liệu đã xử lý (Cleaned)` của file `Mushroom_Preprocessed_Dataset_Cleaned.xlsx`; có thể mở trực tiếp trên Google Colab.

---

## 7. Kết quả chính

### Phân phối nhãn & mức độ liên hệ với `class`
- Hai lớp khá cân bằng: **51,8% ăn được – 48,2% độc**.
- Theo Chi-square, 5 thuộc tính liên hệ mạnh nhất với `class` là: `odor` (χ² ≈ 7.660), `spore-print-color` (≈ 4.602), `gill-color` (≈ 3.766), `ring-type` (≈ 2.957) và `stalk-surface-above-ring` (≈ 2.808).
- Với `odor`: nấm có mùi hạnh nhân (`a`) hoặc hồi (`l`) đều ăn được; nấm có mùi hôi (`f`), tanh (`y`), cay (`s`), hăng (`p`), creosote (`c`), ẩm mốc (`m`) đều độc; nấm không mùi (`n`) gồm 3.408 ăn được và 120 độc.

### So sánh Apriori và FP-Growth
Hai thuật toán cho **cùng tập phổ biến và cùng bộ luật** ở cả 9 cấu hình.

| minsup | Số tập phổ biến | Số luật (minconf = 0,6 / 0,7 / 0,8) |
|---|---|---|
| 0,3 | 1.367 | 22.602 / 18.876 / 14.060 |
| 0,4 | 282 | 1.421 / 1.177 / 1.003 |
| 0,5 | 76 | 259 / 192 / 182 |

Thời gian chạy phụ thuộc môi trường và cấu hình thư viện nên **thay đổi giữa các lần chạy** (xem `results/comparison/` và `results/figures/so_sanh_Apriori_FPGrowth.png`); hãy chạy lại trên máy của bạn để có số liệu so sánh.

### Lọc luật (minsup = 0,3, minconf = 0,7)
18.876 luật ban đầu → 15.582 (bỏ lift ≈ 1) → 6.636 (bỏ dư thừa) → **993** (bỏ trùng tập).

### Một số luật tiêu biểu
| Luật | Support | Confidence | Lift |
|---|---|---|---|
| `odor=n` ∧ `ring-number=o` → `class=e` | 0,3545 | 0,9836 | 1,899 |
| `odor=n` ∧ `stalk-shape=t` → `class=e` | 0,3072 | 1,0000 | 1,931 |
| `bruises=f` ∧ `gill-attachment=f` ∧ `gill-spacing=c` ∧ `ring-number=o` → `class=p` | 0,3880 | 0,9563 | 1,984 |
| `bruises=f` ∧ `gill-spacing=c` ∧ `population=v` ∧ `veil-color=w` → `class=p` | 0,3023 | 0,9715 | 2,015 |

*(Các ký hiệu tra cứu ở mục 3 hoặc trong `docs/value_mapping.csv`.)*

> 📝 `results/rules/Mushroom_Rules_Optimal.xlsx` hiện **chưa chứa luật nào** (các sheet chỉ có 1 tập phổ biến và 0 luật). Nên chạy lại bước xuất file này trước khi dùng. Các luật tiêu biểu ở trên lấy từ `results/comparison/Tuan2_Apriori_vs_FPGrowth_Ketqua.xlsx`.

---

## 8. Giới hạn & cảnh báo

- Tương quan trong luật kết hợp **không phải quan hệ nhân quả**.
- Luật chỉ đúng trên tập dữ liệu này; confidence cao không đồng nghĩa nấm an toàn.
- Mùi, màu sắc và các đặc điểm khác được mô tả chủ quan khi quan sát ngoài đời thực.
- **Tuyệt đối không dùng dự án này làm căn cứ hái hoặc ăn nấm.** Hãy hỏi chuyên gia nấm học địa phương; nếu nghi ngờ ngộ độc, cần đến cơ sở y tế ngay.

## 9. Tài liệu tham khảo
- UCI Machine Learning Repository — *Mushroom Data Set*.
- Kaggle — *Mushroom Classification*.
- Agrawal & Srikant (1994), *Fast Algorithms for Mining Association Rules* (Apriori); Han, Pei & Yin (2000), *Mining Frequent Patterns without Candidate Generation* (FP-Growth).
