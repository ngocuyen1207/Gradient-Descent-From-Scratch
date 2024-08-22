## Optimizers

### 1. Gradient Descent Optimizer
**Khái niệm**: Gradient Descent là một thuật toán tối ưu hóa để tìm giá trị cực tiểu của hàm số. Nó cập nhật các tham số của mô hình bằng cách di chuyển theo hướng ngược lại của gradient của hàm mất mát.

**Công thức**:
\[ \theta = \theta - \eta \nabla J(\theta) \]
Trong đó:
- \(\theta\) là các tham số của mô hình.
- \(\eta\) là learning rate.
- \(\nabla J(\theta)\) là gradient của hàm mất mát.

**Các bước thực hiện**:
1. Khởi tạo các tham số của mô hình.
2. Tính gradient của hàm mất mát.
3. Cập nhật các tham số theo hướng ngược lại của gradient.
4. Lặp lại cho đến khi hội tụ.

### 2. Mini-Batch Gradient Descent Optimizer
**Khái niệm**: Mini-Batch Gradient Descent là một biến thể của Gradient Descent, trong đó gradient được tính trên một tập con nhỏ (mini-batch) của dữ liệu thay vì toàn bộ dữ liệu.

**Công thức**:
\[ \theta = \theta - \eta \nabla J_{mini-batch}(\theta) \]

**Các bước thực hiện**:
1. Chia dữ liệu thành các mini-batch.
2. Khởi tạo các tham số của mô hình.
3. Với mỗi mini-batch, tính gradient của hàm mất mát.
4. Cập nhật các tham số theo hướng ngược lại của gradient.
5. Lặp lại cho đến khi hội tụ.

### 3. Momentum Optimizer
**Khái niệm**: Momentum là một kỹ thuật giúp tăng tốc độ hội tụ của Gradient Descent bằng cách thêm một động lượng vào quá trình cập nhật tham số.

**Công thức**:
\[ v = \gamma v + \eta \nabla J(\theta) \]
\[ \theta = \theta - v \]
Trong đó:
- \(v\) là động lượng.
- \(\gamma\) là hệ số động lượng.

**Các bước thực hiện**:
1. Khởi tạo các tham số và động lượng.
2. Tính gradient của hàm mất mát.
3. Cập nhật động lượng.
4. Cập nhật các tham số theo động lượng.
5. Lặp lại cho đến khi hội tụ.

### 4. RMSprop Optimizer
**Khái niệm**: RMSprop là một thuật toán tối ưu hóa giúp điều chỉnh learning rate cho từng tham số dựa trên giá trị trung bình của gradient bình phương.

**Công thức**:
\[ E[g^2]_t = \gamma E[g^2]_{t-1} + (1 - \gamma) g_t^2 \]
\[ \theta = \theta - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t \]
Trong đó:
- \(E[g^2]_t\) là giá trị trung bình của gradient bình phương.
- \(\epsilon\) là một giá trị nhỏ để tránh chia cho 0.

**Các bước thực hiện**:
1. Khởi tạo các tham số và giá trị trung bình của gradient bình phương.
2. Tính gradient của hàm mất mát.
3. Cập nhật giá trị trung bình của gradient bình phương.
4. Cập nhật các tham số.
5. Lặp lại cho đến khi hội tụ.

### 5. Adam Optimizer
**Khái niệm**: Adam (Adaptive Moment Estimation) là một thuật toán tối ưu hóa kết hợp giữa Momentum và RMSprop.

**Công thức**:
\[ m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \]
\[ v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \]
\[ \hat{m}_t = \frac{m_t}{1 - \beta_1^t} \]
\[ \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \]
\[ \theta = \theta - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t \]
Trong đó:
- \(m_t\) và \(v_t\) là các giá trị trung bình của gradient và gradient bình phương.
- \(\beta_1\) và \(\beta_2\) là các hệ số trung bình.

**Các bước thực hiện**:
1. Khởi tạo các tham số và các giá trị trung bình.
2. Tính gradient của hàm mất mát.
3. Cập nhật các giá trị trung bình.
4. Cập nhật các tham số.
5. Lặp lại cho đến khi hội tụ.

### 6. AdaGrad Optimizer
**Khái niệm**: AdaGrad (Adaptive Gradient) là một thuật toán tối ưu hóa điều chỉnh learning rate cho từng tham số dựa trên tổng bình phương của gradient.

**Công thức**:
\[ G_t = G_{t-1} + g_t^2 \]
\[ \theta = \theta - \frac{\eta}{\sqrt{G_t} + \epsilon} g_t \]

**Các bước thực hiện**:
1. Khởi tạo các tham số và tổng bình phương của gradient.
2. Tính gradient của hàm mất mát.
3. Cập nhật tổng bình phương của gradient.
4. Cập nhật các tham số.
5. Lặp lại cho đến khi hội tụ.

### 7. AdamW Optimizer
**Khái niệm**: AdamW là một biến thể của Adam, trong đó có thêm thuật toán Weight Decay để tránh overfitting.

**Công thức**:
\[ m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \]
\[ v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \]
\[ \hat{m}_t = \frac{m_t}{1 - \beta_1^t} \]
\[ \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \]
\[ \theta = \theta - \eta (\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta) \]
Trong đó:
- \(\lambda\) là hệ số Weight Decay.

**Các bước thực hiện**:
1. Khởi tạo các tham số và các giá trị trung bình.
2. Tính gradient của hàm mất mát.
3. Cập nhật các giá trị trung bình.
4. Cập nhật các tham số với Weight Decay.
5. Lặp lại cho đến khi hội tụ.

### 8. Nadam Optimizer
**Khái niệm**: Nadam (Nesterov-accelerated Adaptive Moment Estimation) là một biến thể của Adam, trong đó có thêm thuật toán Nesterov Momentum.

**Công thức**:
\[ m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \]
\[ v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \]
\[ \hat{m}_t = \frac{m_t}{1 - \beta_1^t} \]
\[ \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \]
\[ \theta = \theta - \eta (\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \beta_1 \frac{g_t}{\sqrt{\hat{v}_t} + \epsilon}) \]

**Các bước thực hiện**:
1. Khởi tạo các tham số và các giá trị trung bình.
2. Tính gradient của hàm mất mát.
3. Cập nhật các giá trị trung bình.
4. Cập nhật các tham số với Nesterov Momentum.
5. Lặp lại cho đến khi hội tụ.

### 9. Adadelta Optimizer
**Khái niệm**: Adadelta là một biến thể của AdaGrad, trong đó có thêm thuật toán để điều chỉnh learning rate dựa trên giá trị trung bình của gradient bình phương.

**Công thức**:
\[ E[g^2]_t = \rho E[g^2]_{t-1} + (1 - \rho) g_t^2 \]
\[ \Delta \theta_t = - \frac{\sqrt{E[\Delta \theta^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} g_t \]
\[ E[\Delta \theta^2]_t = \rho E[\Delta \theta^2]_{t-1} + (1 - \rho) (\Delta \theta_t)^2 \]
\[ \theta = \theta + \Delta \theta_t \]

**Các bước thực hiện**:
1. Khởi tạo các tham số và các giá trị trung bình.
2. Tính gradient của hàm mất mát.
3. Cập nhật các giá trị trung bình của gradient bình phương.
4. Cập nhật các tham số.
5. Lặp lại cho đến khi hội tụ.

### 10. Nesterov Optimizer
**Khái niệm**: Nesterov Momentum là một biến thể của Momentum, trong đó gradient được tính tại vị trí dự đoán của tham số.

**Công thức**:
\[ v = \gamma v + \eta \nabla J(\theta - \gamma v) \]
\[ \theta = \theta - v \]

**Các bước thực hiện**:
1. Khởi tạo các tham số và động lượng.
2. Tính gradient của hàm mất mát tại vị trí dự đoán.
3. Cập nhật động lượng.
4. Cập nhật các tham số theo động lượng.
5. Lặp lại cho đến khi hội tụ.