import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

class GSMSimulator:
    def __init__(self, datalen = 148):
        # GMS 時槽長度為 156.25 μs，包含 148 bits 的資料和 8 bits 的尾碼
        self.datalen = datalen
        self.tail_len = 8

    def channel_encoding(self, bits):
        # 使用標準 1/2 卷積碼進行編碼
        state = [0,0]
        encode = [] 
        for bit in bits: 
            out1 = bit ^ state[0] ^ state[1]  # 第一個輸出
            out2 = bit ^ state[1]  # 第二個輸出
            encode.extend([out1, out2]) # 將兩個輸出加入編碼結果
            state = [bit, state[0]] # 更新狀態
        return np.array(encode)
    
    def interleaving(self, bits):
        # 簡單的交織器，將資料重新排列，打散連續的錯誤
        return bits.reshape(-1, 4).T.flatten()
    
    def gmsk_modulation(self, bits):
        """GMSK 調變: 2G 的核心，為了讓訊號包絡穩定並節省頻寬"""
        # 這裡簡化為二進制相位調變，實際 GMSK 更複雜
        return 2 * bits - 1  # 將 0 和 1 映射到 -1 和 +1
    
    def awgn_channel(self, signal, snr_db):
        """添加高斯白噪聲""" # 將 SNR 從 dB 轉換為線性比例
        snr_linear = 10 ** (snr_db / 10)
        noise_std = np.sqrt(1 / (2 * snr_linear))  # 計算噪聲標準差
        noise = np.random.normal(0, noise_std, signal.shape)  # 生成高斯白噪聲
        return signal + noise

    def deinterleaving(self, bits):
        # 反交織器，將資料恢復到原來的順序
        return bits.reshape(4, -1).T.flatten()

    def viterbi_decoding(self, bits):
        hard_decision = (bits > 0).astype(int)  # 將接收的信號轉換為二進制
        return hard_decision[::2]  # 取出第一個輸出，簡化解碼過程
    
    def run_simulate(self, snr_db = 5):
        tx_bits = np.random.randint(0, 2, self.datalen)  # 生成隨機資料

        c_bits = self.channel_encoding(tx_bits)  # 編碼
        i_bits = self.interleaving(c_bits)  # 交織
        mod_signal = self.gmsk_modulation(i_bits)  # 調變

        rx_signal = self.awgn_channel(mod_signal, snr_db)  # 通過 AWGN 信道

        deint_bits = self.deinterleaving(rx_signal)  # 反交織
        rx_bits = self.viterbi_decoding(deint_bits)  # 解碼

        ber = np.mean(tx_bits != rx_bits)  # 計算位元錯誤率
        return tx_bits, rx_bits, ber

sim = GSMSimulator(datalen = 200)
tx, rx, ber = sim.run_simulate()
print(f"--- 2G 通訊模擬報告 ---")
print(f"原始資料 (前20位): {tx[:20]}")
print(f"解碼資料 (前20位): {rx[:20]}")
print(f"目前誤碼率 (BER): {ber:.4f}")

# 繪製訊號圖形，這對工程師診斷系統非常重要
plt.figure(figsize=(10, 4))
plt.step(range(50), tx[:50], label='Original Bits', where='post')
plt.step(range(50), rx[:50], label='Recovered Bits', where='post', linestyle='--')
plt.title(f"Signal Recovery (SNR={7}dB, BER={ber})")
plt.legend()