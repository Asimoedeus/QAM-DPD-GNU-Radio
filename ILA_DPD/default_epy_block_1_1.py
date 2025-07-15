import numpy as np
from gnuradio import gr

class PA_model_GMP(gr.sync_block):
    """
    Generalized Memory Polynomial (GMP) PA Model Block.

    This block implements a GMP model defined by the equation:
    y[n] = sum_{k=1 to K} sum_{q=0 to Q} a_kq * x[n-q] * |x[n-q]|^{k-1}

    The coefficients 'a_kq' are provided as a 2D list or numpy array.
    """
    def __init__(self, coeffs= [
    [0.95+0.01j, 0.05-0.02j],  # k=1: a_10, a_11 (线性项和记忆项)
    [0.00-0.00j, 0.00+0.00j],  # k=2: a_20, a_21 (主要引起AM/PM)
    [-0.1-0.08j, -0.02-0.01j]  # k=3: a_30, a_31 (主要引起AM/AM)
]):
        """
        Args:
            coeffs (list or numpy.ndarray): A 2D array-like structure for the
                complex coefficients a_kq. The shape should be K x (Q+1).
                Example: [[a_10, a_11], [a_20, a_21], [a_30, a_31]]
                for K=3, Q=1.
        """
        gr.sync_block.__init__(
            self,
            name='PA_model_GMP',
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )

        # --- 从系数矩阵中推导模型参数 ---
        self.coeffs = np.array(coeffs, dtype=np.complex64)
        if self.coeffs.ndim != 2:
            raise ValueError("Coefficients must be a 2D array-like object.")

        # K is the number of rows (non-linearity order)
        # Q is (number of columns - 1) (memory depth)
        self.K = self.coeffs.shape[0]
        self.Q = self.coeffs.shape[1] - 1
        
        if self.Q < 0:
            raise ValueError("Coefficients matrix must have at least one column.")

        # --- 初始化用于存储历史样本的状态变量 ---
        # self.history stores the last Q input samples for memory effects.
        self.history = np.zeros(self.Q, dtype=np.complex64)

    def work(self, input_items, output_items):
        # input_items[0] is the new block of input samples
        # output_items[0] is where we write our output samples
        x_new = input_items[0]
        y_out = output_items[0]
        
        # --- 核心处理逻辑 ---

        # 1. 将历史样本与新输入样本拼接，形成一个完整的工作向量
        #    This allows us to compute y[n] which depends on x[n-q]
        x_full = np.concatenate((self.history, x_new))
        
        # 获取新输入的长度
        num_new_samples = len(x_new)
        
        # 初始化输出为零
        # The output is a summation, so we start with zeros.
        y_out[:] = 0.0

        # 2. 执行GMP双重求和计算
        # Outer loop over non-linearity order k (from 1 to K)
        for k_idx in range(self.K):
            k = k_idx + 1 # k runs from 1 to K
            # Inner loop over memory depth q (from 0 to Q)
            for q in range(self.Q + 1):
                # 获取系数 a_kq
                a_kq = self.coeffs[k_idx, q]
                
                # 获取带有延迟的输入信号 x[n-q]
                # For an output block of size N, the corresponding input
                # slice from x_full is x_full[Q-q : Q-q+N]
                delayed_input = x_full[self.Q - q : self.Q - q + num_new_samples]
                
                # 计算当前项并累加到输出
                # term = a_kq * x[n-q] * |x[n-q]|^{k-1}
                term = a_kq * delayed_input * (np.abs(delayed_input)**(k - 1))
                y_out[:] += term

        # 3. 更新历史记录，为下一次调用work做好准备
        # The last Q samples of the full input vector become the new history.
        self.history = x_full[-self.Q:] if self.Q > 0 else np.array([])
        
        # 4. 返回处理的样本数
        return len(y_out)