---
series: "3B1B - Neural Networks 神經網路"
episode: 1
date: 2026-05-18
tags:
  - 3b1b
  - neural-networks
  - learning-note
status: complete
---

# Ep.01 — But what is a neural network? 什麼是神經網路？

> **影片連結 Video Link**：[YouTube](https://www.youtube.com/watch?v=aircAruvnKk)

## 一句話總結 One-Line Summary

A neural network is a massively complex mathematical function — it takes in 784 numbers and spits out 10, with ~13,000 learnable parameters in between.  
神經網路就是一個極度複雜的數學函數——吃進 784 個 pixel 紀錄的數字，吐出 10 個最終預測(0~9)，中間藏著約 13,000 個可學習的參數。

## 關鍵直覺 Key Intuitions

- **抽象化是層層堆疊的 Abstraction is hierarchical**：從像素到邊緣到筆畫到數字，每一層隱藏層都在做「把低階特徵組合成高階意義」這件事。  
  Pixel → Edge → Pattern → Digit — each hidden layer assembles lower-level features into higher-level meaning.

- **神經元是函數，不是容器 A neuron is a function, not a container**：更精確的理解是，每個神經元是一個函數，輸入前一層的資料，輸出一個數值給下一層。整個網路是一個超級複合函數。  
  Each neuron is a function that takes in data from the previous layer and outputs a value to the next — the whole network is one giant composite function.

- **一條連線就是一個信任程度 One connection = one trust level**：每條連線回答的問題是「我應該聽前面那個神經元的話聽多少？」，這個「多少」就是權重。  
  Each connection answers: "How much should I trust the neuron on the other end?" That "how much" is the weight.

- **「學習」的真面目 What "learning" really means**：機器學習不是什麼魔法，就是讓電腦自己去找出那 13,000 個權重與偏置的最佳設定值。  
  ML is not magic — it's just finding the optimal values for ~13,000 weights and biases.

## 重要術語 Key Terms

| 中文 | English | 說明 Description |
|------|---------|-----------------|
| 神經元 | Neuron | 網路的基本單位，裝著一個 0~1 的激勵值 / Basic unit holding an activation value between 0 and 1 |
| 激勵值 | Activation | 神經元持有的數值，代表「激發程度」/ The number a neuron holds, representing how activated it is |
| 權重 | Weight | 神經元間連線的強度，代表對前一個神經元的「信任程度」/ Strength of connection — how much to trust the previous neuron |
| 偏置 | Bias | 激發的門檻值，加權總和須超過它才會點亮 / Threshold for activation — weighted sum must exceed it to fire |
| 輸入層 | Input Layer | 784 個神經元，對應 28×28 像素 / 784 neurons corresponding to 28×28 pixels |
| 隱藏層 | Hidden Layer | 負責拆解與抽象化資訊 / Responsible for decomposing and abstracting information |
| 輸出層 | Output Layer | 10 個神經元，對應數字 0~9 / 10 neurons corresponding to digits 0–9 |
| Sigmoid 函數 | Sigmoid Function | 將任意實數壓縮到 0~1 的函數，早期常用 / Squishes any real number into 0–1 range, used in early networks |
| ReLU | ReLU (Rectified Linear Unit) | 現代主流激勵函數，小於 0 輸出 0，大於 0 等於輸入值 / Modern standard: outputs 0 if input < 0, else outputs input |

## 筆記 Notes

### 網路結構 Network Structure

```
輸入層 Input (784)  →  隱藏層 Hidden (16)  →  隱藏層 Hidden (16)  →  輸出層 Output (10)
      像素 pixels         邊緣 edges             圖樣 patterns          數字 digits
```

### 為什麼一條連線就是一個權重？Why does one connection = one weight?

一條連線的任務，就是回答「我應該聽前面那個神經元的話聽多少？」這個「多少」就是權重。  
Each connection answers one question: "How much should I trust the neuron on the other end?" That "how much" is the weight.

以隱藏層中負責「偵測左上角邊緣」的神經元為例：  
Take a hidden neuron whose job is to detect edges in the top-left corner:

- 左上角的像素 → 非常在乎 → 權重設很大  
  Top-left pixels → highly relevant → large weight
- 右下角的像素 → 無關 → 權重接近 0  
  Bottom-right pixels → irrelevant → weight near 0
- 某些像素反而是反指標 → 權重設成負數  
  Some pixels are counter-indicators → negative weight

所以這個神經元的輸出是：  
So this neuron computes:

$$\text{output} = w_1 a_1 + w_2 a_2 + \dots + w_{784} a_{784} + b$$

每一個 $w$ 就是對那條連線的「信任程度」。有連線才有信任程度，所以**一條連線 = 一個權重**。  
Each $w$ is the neuron's trust level for that connection. No connection, no trust level — so **one connection = one weight**.

### 參數量計算 Parameter Count

第二層 16 個神經元，每個神經元的任務不同（偵測不同的特徵），所以每個人對 784 個輸入的信任程度都不一樣，各自擁有獨立的 784 個權重。  
The 16 neurons in the second layer each have different jobs, so each has its own independent set of 784 trust levels.

$$16 \text{ 個神經元} \times 784 \text{ 個權重} = 784 \times 16 = 12{,}544$$

完整計算 Full parameter count：

$$\underbrace{784 \times 16}_{12{,}544} + \underbrace{16 \times 16}_{256} + \underbrace{16 \times 10}_{160} + \underbrace{16 + 16 + 10}_{42 \text{ biases}} = 13{,}002$$

### 矩陣化加速 Matrix Formulation

單一神經元的計算寫法冗長：  
Computing a single neuron longhand is verbose:

$$\sigma(w_{0,0}a_0 + w_{0,1}a_1 + \dots + b)$$

矩陣化後變得簡潔，且適合 GPU 平行運算：  
Matrix form is compact and GPU-parallelizable:

$$\sigma(\mathbf{W}\mathbf{a} + \mathbf{b})$$

這就是為什麼 AI 需要 GPU — GPU 本質上就是大量平行矩陣運算的硬體。  
This is why AI needs GPUs — GPUs are hardware built for massively parallel matrix operations.

### Sigmoid vs. ReLU

| | Sigmoid | ReLU |
|---|---|---|
| 輸出範圍 Output range | 0 ~ 1 | 0 ~ ∞ |
| 早期用途 Early use | 模仿神經元開/關 Mimic neuron on/off | — |
| 問題 Problem | 深層網路梯度消失 Vanishing gradient in deep nets | Dying ReLU（神經元永久死掉） |
| 現代主流 Modern standard | ❌ | ✅ |

## 跟我的關聯 Relevance to Me

數位孿生系統裡的物件辨識（例如 Isaac Sim 的感測器資料處理）本質上也是在做類似的事——把感測器原始數值（像素、點雲）透過層層抽象化轉換成有意義的判斷。矩陣乘法 $\mathbf{Wa} + \mathbf{b}$ 的形式跟座標轉換矩陣很像，只是這裡的矩陣是學出來的，不是手動設計的。

Object recognition in digital twin systems (e.g., sensor data in Isaac Sim) does essentially the same thing — transforming raw sensor values (pixels, point clouds) into meaningful decisions through layered abstraction. The matrix form $\mathbf{Wa} + \mathbf{b}$ resembles coordinate transformation matrices, except here the matrices are learned rather than manually designed.

## 還沒搞懂的 Still Unclear

- [ ] 隱藏層「理想上」負責辨識邊緣和筆畫，但實際訓練出來的神經元真的是這樣分工的嗎？還是完全不同的東西？  
      Ideally hidden layers detect edges then patterns — but do trained networks actually work this way, or do neurons learn something completely different?

## 連結 Links

[[梯度下降 Gradient Descent]] · [[激活函數 Activation Function]] · [[矩陣 Matrix]] · [[向量 Vector]]
