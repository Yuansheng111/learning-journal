---
series: "3B1B - Neural Networks 神經網路"
episode: 3
date: 2026-05-19
tags:
  - 3b1b
  - neural-networks
  - learning-note
status: complete
---

# Ep.03 — Backpropagation, intuitively 反向傳播，直覺理解

> **影片連結 Video Link**：[YouTube](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

## 一句話總結 One-Line Summary

Backpropagation is an accountability algorithm — when the output is wrong, it traces back layer by layer and calculates exactly how much each weight and bias is responsible, then tells them how to adjust.  
反向傳播是一套「問責與分配」的演算法——當輸出出錯時，從結果端一層層往回追溯，精準算出每個權重和偏置應該為錯誤負多少責任，並指示它們該如何微調。

## 關鍵直覺 Key Intuitions

- **梯度的大小代表相對重要性 Gradient magnitude = relative importance**：梯度向量裡的每個數值代表成本對該參數的敏感度。調整權重 A 能讓成本降 3 單位、調整 B 只降 0.1 單位，那現在 A 遠比 B 重要。反向傳播就是在找這些高效益的關鍵參數。  
  Each value in the gradient vector represents how sensitive the cost is to that parameter. If adjusting weight A drops cost by 3 and weight B by 0.1, then A is 30x more valuable right now. Backpropagation finds these high-leverage parameters.

- **神經元有「渴望」A neuron has desires**：輸入一張「2」的圖片，輸出層代表「2」的神經元渴望自己變亮，其他 9 個渴望變暗。這些「渴望」就是驅動反向傳播的訊號源頭。  
  Feed in an image of "2" — the output neuron for "2" wants to be brighter, the other 9 want to dim. These "desires" are the signals that drive backpropagation.

- **為什麼叫「反向」Why "back" propagation**：因為訊號從輸出層開始，一層一層往輸入層方向倒推。輸出層的願望傳給倒數第二層，再傳給倒數第三層……直到最前面。  
  Because signals flow from the output layer backward toward the input layer. Output layer desires propagate to the second-to-last layer, then to the third-to-last… all the way to the front.

- **赫布理論的迴響 Hebbian Theory in action**：「一起發火的神經元，會連在一起」。上一層的神經元越亮、與它相連的權重增加效益越大——網路在學習過程中，自然對「共同激發的模式」加強連結。  
  "Neurons that fire together, wire together." The brighter the previous neuron, the more impactful it is to increase the connecting weight — the network naturally strengthens links between co-activated patterns.

- **SGD 是跌跌撞撞的醉漢 SGD is a stumbling drunk**：每次只看一小批資料（Mini-batch）算出的梯度不是最完美的下坡方向，軌跡像醉漢——但腳步快，最終仍能抵達谷底。  
  Gradients computed from a mini-batch aren't perfectly downhill — the path looks like a stumbling drunk. But the steps are fast, and the drunk still reaches the valley.

## 重要術語 Key Terms

| 中文 | English | 說明 Description |
|------|---------|-----------------|
| 敏感度 | Sensitivity | 成本函數對某個參數的變化有多敏感，即梯度的大小 / How much the cost changes when a parameter changes — i.e., gradient magnitude |
| 反向傳播 | Backpropagation | 從輸出層往回逐層計算梯度的演算法 / Algorithm that computes gradients layer by layer from output back to input |
| 願望清單 | Desired changes | 每個神經元對前一層參數期望的修改方向與幅度 / Each neuron's expected direction and magnitude of change for the previous layer's parameters |
| 赫布理論 | Hebbian Theory | 「一起發火的神經元，會連在一起」/ "Neurons that fire together, wire together" |
| 隨機梯度下降 | Stochastic Gradient Descent (SGD) | 每次只用一小批資料估算梯度並更新參數的方法 / Method that estimates gradients from a small batch of data and updates parameters each step |
| 微批次 | Mini-batch | 訓練資料中隨機抽取的一小批樣本，例如 100 張 / A small randomly sampled subset of training data, e.g., 100 images |

## 筆記 Notes

### 讓神經元變亮的三種途徑 Three Ways to Increase a Neuron's Activation

以輸出層「2」的神經元為例，想讓它變亮：  
For the output neuron representing "2" — to make it brighter:

| 途徑 Method | 說明 Description | 效益最大化條件 When most effective |
|---|---|---|
| 增加偏置 Increase bias | 直接拉高激發門檻 / Directly raises the activation threshold | 永遠有效 Always effective |
| 增加權重 Increase weights | 加強與前一層的連線強度 / Strengthen connections from previous layer | 前一層神經元越亮效益越大 / More effective when previous neurons are already bright |
| 改變前一層激勵值 Change previous activations | 讓正權重連線的神經元變亮、負權重的變暗 / Brighten neurons with positive weights, dim those with negative weights | 權重絕對值越大效益越高 / More effective when weight magnitude is large |

### 反向傳播的完整流程 Full Backpropagation Flow

```
輸出層 Output layer
  → 每個神經元產生「願望清單」Each neuron generates desired changes
  → 10 個神經元的願望加總 Sum desires from all 10 output neurons
  → 得出對倒數第二層的修改清單 Get update list for second-to-last layer
  → 用同樣邏輯繼續往前推 Apply same logic to previous layer
  → 一路推到輸入層 Continue all the way to the input layer
```

關鍵：我們無法直接修改中間層的激勵值（那是由更前面的層決定的），所以只能把「期望」記錄下來往回傳遞。  
Key: we can't directly change intermediate layer activations (they're determined by earlier layers), so we record the "desires" and pass them backward.

### 從單一樣本到整體：SGD From One Sample to All: SGD

|                         | 理想梯度下降 Ideal GD             | 隨機梯度下降 SGD                                                |
| ----------------------- | --------------------------- | --------------------------------------------------------- |
| 每步看幾張資料 Data per step   | 全部幾萬張 All tens of thousands | 一個 Mini-batch（如 100 張）/ One mini-batch (e.g., 100 images) |
| 梯度準確度 Gradient accuracy | 完美 Perfect                  | 近似，有噪音 Approximate, noisy                                 |
| 計算速度 Speed              | 極慢 Very slow                | 極快 Very fast                                              |
| 下山軌跡 Path shape         | 平滑 Smooth                   | 跌跌撞撞 Stumbling                                            |
| 最終結果 Final result       | 收斂到谷底 Converges to valley   | 也能收斂到谷底 Also converges to valley                          |

### 為什麼只看一張圖不夠？Why One Image Isn't Enough?

如果只根據一張「2」的圖片更新參數，網路最終只會把所有東西都認成 2。真正的梯度必須整合所有訓練資料的期望——SGD 是用 Mini-batch 近似這個整合。  
If we only update based on one "2" image, the network will eventually classify everything as 2. True gradient must integrate all training data — SGD approximates this with mini-batches.

## 跟我的關聯 Relevance to Me

在自動化配管專案裡，處理的是層層包裹的 Rule-based 函數——最終輸出某個配管細節，但有時候結果是錯的。這時候要做的追查邏輯跟反向傳播很像：從錯誤的結果往回追查，穿過一層又一層的函數邏輯，評估每一層對這個錯誤要負多少責任。

不過這裡有一個關鍵差異值得記下來：配管除錯時，通常找到根源參數就修那一個。但反向傳播是**每一層的參數都調整**，只是幅度按各自的責任大小分配——梯度大的調多，梯度小的調少，沒有哪一層被跳過。這也是為什麼下一集需要鏈式法則（Chain Rule）：要算出每一層的責任，而不是只找到最底層的那個根源。

In the automated piping project, the system is built from layers of nested rule-based functions — the final output is a piping detail, but sometimes it's wrong. The tracing logic resembles backpropagation: follow the error back through layer after layer of function logic, assessing how much each layer contributed to the mistake.

But there's a key difference worth noting: when debugging piping, we typically find the root parameter and fix just that one. Backpropagation adjusts parameters at **every layer simultaneously** — with magnitudes proportional to each layer's responsibility. Larger gradient means larger adjustment; no layer is skipped. This is also why the chain rule is needed next episode: to compute each layer's responsibility, rather than tracing down to a single root cause.

## 還沒搞懂的 Still Unclear

- [ ] 鏈式法則（Chain Rule）具體是怎麼計算每一層的梯度？反向傳播的實際數學推導是什麼樣子？（下一集）  
      How exactly does the chain rule compute gradients at each layer? What does the actual math of backpropagation look like? (Next episode)
- [ ] Mini-batch 的大小怎麼選？太小或太大各有什麼問題？  
      How do you choose mini-batch size? What are the tradeoffs of too small vs. too large?
- [x] ~~SGD 的「隨機 (Stochastic)」指的是資料的隨機抽取，還是別的隨機性？~~ → 這集已解答：指的是隨機抽取 Mini-batch  
      ~~Does "stochastic" in SGD refer to the random sampling of data, or some other source of randomness?~~ → Answered: refers to random mini-batch sampling

## 連結 Links

[[反向傳播 Backpropagation]] · [[梯度下降 Gradient Descent]] · [[成本函數 Cost Function]] · [[激活函數 Activation Function]] · [[隨機梯度下降 SGD]]
