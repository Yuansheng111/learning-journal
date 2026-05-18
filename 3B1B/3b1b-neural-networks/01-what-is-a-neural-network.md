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
- Pixel → Edge → Pattern → Digit — each hidden layer assembles lower-level features into higher-level meaning.

- **神經元是函數，不是容器 A neuron is a function, not a container**：
- 更精確的理解是，每個神經元是一個函數，輸入前一層的資料，輸出一個數值給下一層。整個網路是一個超級複合函數。Each neuron is a function that takes in data from the previous layer and outputs a value to the next — the whole network is one giant composite function.

- **「學習」的真面目 What "learning" really means**：機器學習不是什麼魔法，就是讓電腦自己去找出那 13,000 個權重與偏置的最佳設定值。ML is not magic — it's just finding the optimal values for ~13,000 weights and biases.

## 重要術語 Key Terms

| 中文         | English                      | 說明 Description                                                                                      |
| ---------- | ---------------------------- | --------------------------------------------------------------------------------------------------- |
| 神經元        | Neuron                       | 網路的基本單位，裝著一個 0~1 的激勵值 <br>Basic unit holding an activation value between 0 and 1                    |
| 激勵值        | Activation                   | 神經元持有的數值，代表「激發程度」<br>The number a neuron holds, representing how activated it is                    |
| 權重         | Weight                       | 神經元間連線的強度，決定關注哪種圖樣 <br>Strength of connections between neurons, determines what pattern to look for |
| 偏置         | Bias                         | 激發的門檻值，加權總和須超過它才會點亮 <br>Threshold for activation — weighted sum must exceed it to fire              |
| 輸入層        | Input Layer                  | 784 個神經元，對應 28×28 像素 <br>784 neurons corresponding to 28×28 pixels                                  |
| 隱藏層        | Hidden Layer                 | 負責拆解與抽象化資訊 <br>Responsible for decomposing and abstracting information                              |
| 輸出層        | Output Layer                 | 10 個神經元，對應數字 0~9 <br>10 neurons corresponding to digits 0–9                                         |
| Sigmoid 函數 | Sigmoid Function             | 將任意實數壓縮到 0~1 的函數，早期常用 <br>Squishes any real number into 0–1 range, used in early networks           |
| ReLU       | ReLU (Rectified Linear Unit) | 現代主流激勵函數，小於 0 輸出 0，大於 0 等於輸入值<br>Modern standard: outputs 0 if input < 0, else outputs input        |

## 筆記 Notes

### 網路結構 Network Structure

```
輸入層 Input (784)  →  隱藏層 Hidden (16)  →  隱藏層 Hidden (16)  →  輸出層 Output (10)
      像素 pixels         邊緣 edges             圖樣 patterns          數字 digits
```

### 參數量計算 Parameter Count

兩層之間的連線數 = 前一層神經元數 × 後一層神經元數  
Number of connections between layers = neurons in layer A × neurons in layer B

$$784 \times 16 = 12{,}544 \text{ 個權重 weights}$$

加上每層的偏置，整個網路約 13,000 個參數。  
Adding biases for each neuron, the total is ~13,000 parameters.

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

|                      | Sigmoid                            | ReLU  |     |
| -------------------- | ---------------------------------- | ----- | --- |
| 輸出範圍 Output range    | 0 ~ 1                              | 0 ~ ∞ |     |
| 早期用途 Early use       | 模仿神經元開/關 Mimic neuron on/off       | —     |     |
| 問題 Problem           | 深層網路訓練慢 Slow to train in deep nets | —     |     |
| 現代主流 Modern standard | ❌                                  | ✅     |     |

## 跟我的關聯 Relevance to Me

數位孿生系統裡的物件辨識（例如 Isaac Sim 的感測器資料處理）本質上也是在做類似的事——把感測器原始數值（像素、點雲）透過層層抽象化轉換成有意義的判斷。矩陣乘法 $\mathbf{Wa} + \mathbf{b}$ 的形式跟座標轉換矩陣很像，只是這裡的矩陣是學出來的，不是手動設計的。

Object recognition in digital twin systems (e.g., sensor data in Isaac Sim) does essentially the same thing — transforming raw sensor values (pixels, point clouds) into meaningful decisions through layered abstraction. The matrix form $\mathbf{Wa} + \mathbf{b}$ resembles coordinate transformation matrices, except here the matrices are learned rather than manually designed.

## 還沒搞懂的 Still Unclear

- [ ] 隱藏層「理想上」負責辨識邊緣和筆畫，但實際訓練出來的神經元真的是這樣分工的嗎？還是完全不同的東西？  
      Ideally hidden layers detect edges then patterns — but do trained networks actually work this way, or do neurons learn something completely different?
- [ ] Sigmoid 在深層網路中訓練慢的原因是什麼？（梯度消失問題？）  
      Why exactly does Sigmoid train slowly in deep networks? (Vanishing gradient problem?)

## 關鍵知識點連結 Key Knowledge Links

[[梯度下降 Gradient Descent]] · [[激活函數 Activation Function]] · [[矩陣 Matrix]] · [[向量 Vector]]
