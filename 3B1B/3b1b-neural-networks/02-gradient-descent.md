---
series: "3B1B - Neural Networks 神經網路"
episode: 2
date: 2026-05-19
tags:
  - 3b1b
  - neural-networks
  - learning-note
status: complete
---

# Ep.02 — Gradient descent, how neural networks learn 梯度下降，神經網路如何學習

> **影片連結 Video Link**：[YouTube](https://www.youtube.com/watch?v=IHZwWFHWa-w)

## 一句話總結 One-Line Summary

Learning is just finding the lowest point of a cost function in 13,000-dimensional space — gradient descent is how you roll downhill without being able to see the terrain.  
學習就是在 13,000 維空間裡找到成本函數的最低點——梯度下降就是你在看不見地形的情況下往山谷滾的方法。

## 關鍵直覺 Key Intuitions

- **學習的本質 The essence of learning**：一開始所有參數都是隨機的，網路的預測是垃圾。「學習」就是反覆調整這 13,000 個數值，讓輸出越來越接近正確答案。  
  All parameters start random — predictions are garbage. "Learning" is iteratively adjusting all 13,000 values until outputs approach correct answers.

- **為什麼要用連續值而非 0/1 Why continuous values instead of 0/1**：如果神經元只輸出 True/False，成本函數的地形會變成階梯狀——平坦處梯度為 0，邊緣梯度無限大，根本沒辦法下山。平滑的激勵函數才能創造平滑的山坡。  
  Binary neurons would make the cost landscape a staircase — zero gradient on flat steps, infinite gradient at edges. Smooth activation functions create smooth hills that gradient descent can actually roll down.

- **梯度不只有方向，大小也很重要 Gradient magnitude matters too**：梯度向量告訴你哪個參數現在最值得調——數值越大，微調它帶來的成本下降效益越高（bang for your buck）。  
  The gradient vector tells you which parameters are worth adjusting right now — larger magnitude means more cost reduction per unit of change.

- **網路學到的不是幾何特徵 Networks don't learn geometry**：把訓練好的權重視覺化後，第一隱藏層找的根本不是邊緣，而是看似隨機的雜訊模式。網路只是在 13,000 維空間裡切出邊界，不是真正「看懂」圖片。  
  Visualizing trained weights reveals the first hidden layer learns noise-like patterns, not edges. The network carves decision boundaries in parameter space — it doesn't truly "understand" images.

## 重要術語 Key Terms

| 中文 | English | 說明 Description |
|------|---------|-----------------|
| 全連接神經網路 | Fully Connected Neural Network (FCNN) / MLP / Dense Network | 每個神經元都與前一層所有神經元相連的架構 / Architecture where every neuron connects to all neurons in the previous layer |
| 成本函數 | Cost Function / Loss Function | 衡量網路預測與正確答案之間誤差的函數，越低越好 / Measures error between predictions and correct answers — lower is better |
| 誤差平方和 | Sum of Squared Errors | 每個輸出神經元的預測值與理想值之差的平方加總 / Sum of squared differences between predicted and ideal output values |
| 梯度 | Gradient | 多變數函數最陡峭上坡方向的向量 / Vector pointing in the direction of steepest ascent of a multivariable function |
| 負梯度 | Negative Gradient | 最陡峭的下坡方向，梯度下降移動的方向 / Direction of steepest descent — the direction gradient descent moves |
| 梯度下降 | Gradient Descent | 反覆計算梯度並往負梯度方向調整參數的最佳化過程 / Iterative process of computing gradients and stepping in the negative gradient direction |
| 學習率 | Learning Rate | 每次梯度下降的步長大小 / Step size taken at each gradient descent iteration |
| 區域最小值 | Local Minimum | 成本函數地形中的一個谷底，不一定是全域最低點 / A valley in the cost landscape — not necessarily the global lowest point |
| 反向傳播 | Backpropagation | 高效計算梯度的演算法，下一集的主題 / Algorithm for efficiently computing gradients — next episode's topic |
| 卷積神經網路 | CNN (Convolutional Neural Network) | 專為圖像設計、真正尋找空間特徵的網路架構 / Network architecture designed for images that actually searches for spatial features |
| 分類器 | Discriminator / Classifier | 只能判斷輸入屬於哪個類別，無法生成圖像 / Can classify inputs but cannot generate images |

## 筆記 Notes

### 訓練的起點 Starting Point

所有權重與偏置初始值完全隨機 → 預測結果是垃圾 → 需要學習。  
All weights and biases start completely random → predictions are garbage → learning is needed.

### 成本函數的計算 How Cost is Computed

給定一張標籤為「3」的圖片，理想輸出是：  
Given an image labeled "3", the ideal output is:

$$\text{輸出層} = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]$$

成本 = 實際輸出與理想輸出之間的誤差平方和：  
Cost = sum of squared differences between actual and ideal outputs:

$$C = \sum_{i=0}^{9}(\text{actual}_i - \text{ideal}_i)^2$$

整體成本 = 所有訓練樣本的成本平均值。  
Overall cost = average cost across all training examples.

### 梯度下降的直覺 Gradient Descent Intuition

從高維到低維的降維思考：  
Reducing high dimensions to something visualizable:

```
1 個參數 → 2D 曲線（找最低點）
2 個參數 → 3D 碗狀曲面（找谷底）
13,000 個參數 → 同樣的數學，只是人腦無法想像

1 parameter  → 2D curve (find the minimum)
2 parameters → 3D bowl surface (find the valley)
13,000 parameters → same math, just unimaginable for human brains
```

流程 Process：

$$\text{計算梯度 Compute gradient} \rightarrow \text{往負梯度方向走一步 Step in negative gradient direction} \rightarrow \text{重複 Repeat}$$

步長與斜率成正比——越靠近谷底斜率越平，步伐自然變小，不會走過頭。  
Step size is proportional to slope — near the valley the slope is gentle, so steps naturally shrink and overshoot is avoided.

### 為什麼不用擔心區域最小值？Why Not Worry About Local Minima?

論文研究指出：在超高維度的神經網路中，各個區域最小值的品質其實差不多好。梯度下降即使卡在某個谷底，那個谷底通常也夠用了。  
Research shows: in very high-dimensional networks, most local minima are about equally good. Even if gradient descent gets stuck in a valley, that valley is usually good enough.

### 網路的真實限制 Real Limitations of This Network

| | 理想 Ideal | 現實 Reality |
|---|---|---|
| 第一隱藏層學到 1st hidden layer learns | 邊緣 Edges | 隨機雜訊圖案 Random noise patterns |
| 面對隨機雜訊輸入 Given random noise input | 困惑 Confused | 極度自信地亂猜 Extremely confident wrong guess |
| 學習機制 Learning mechanism | 理解幾何 Geometric understanding | 切割高維邊界 Carving high-dimensional boundaries |

### 論文彩蛋 Paper Insights

**《Understanding Deep Learning Requires Rethinking Generalization》**
- 神經網路參數多到可以完全死背隨機亂標籤的資料。  
  Networks have enough parameters to memorize completely randomly labeled data.
- 但餵給正確標籤時，成本下降速度遠比亂標籤快。  
  But with correct labels, cost drops much faster than with random labels.
- 結論：網路有死背能力，但天生傾向先學習資料中真實存在的規律。  
  Conclusion: networks can memorize, but naturally prefer to learn genuine patterns first.

## Q&A 延伸釐清

### Q1｜什麼是 MNIST？What is MNIST?

機器學習界的「Hello World」。MNIST 是一個公開資料庫，包含數萬張由高中生和人口普查局員工手寫的數字圖片（0 到 9），每張標準化為 28×28 像素的灰階圖。它是測試與訓練圖像辨識演算法最經典的基準測試。  
The "Hello World" of machine learning. MNIST is a public dataset of tens of thousands of handwritten digit images (0–9), each standardized to a 28×28 grayscale image. It's the most classic benchmark for testing and training image recognition algorithms.

### Q2｜Cost 到底是什麼類型的值？What kind of value is Cost?

Cost 不是參數（如權重 $w$、偏置 $b$），也不是超參數（如網路層數、學習率這些你手動設定的規則）。它是這個龐大函數的**輸出評分指標**——一個純量（Scalar），透過實際輸出與標準答案計算出來，衡量目前這組參數的表現有多爛。評分越低，網路越好。  
Cost is neither a parameter (like weight $w$ or bias $b$), nor a hyperparameter (like number of layers or learning rate, which you set manually). It's the **output score** of the whole system — a scalar computed from actual outputs vs. correct answers, measuring how bad the current parameters are. Lower score = better network.

| 類型 Type             | 例子 Examples                                                          |
| ------------------- | -------------------------------------------------------------------- |
| 參數 Parameters       | 權重 $w$、偏置 $b$                                                        |
| 超參數 Hyperparameters | 網路層數、每層神經元數、學習率 / Number of layers, neurons per layer, learning rate |
| 評分指標 Score metric   | **Cost / Loss** ← 這個                                                 |

### Q3｜什麼是多變量微積分？What is Multivariable Calculus?

高中微積分處理單一變數 $f(x)$，畫出來是一條 2D 曲線，「斜率」就是導數。多變量微積分處理擁有多個輸入變數的函數，例如 $f(x, y, z)$。我們的 Cost 函數有 13,000 個輸入變數，所以需要多變量微積分中的「梯度」概念，才能在超高維度空間裡找出最陡峭的下坡方向。  
High school calculus deals with single-variable functions $f(x)$ — a 2D curve where "slope" is the derivative. Multivariable calculus handles functions with multiple inputs, like $f(x, y, z)$. Our cost function has 13,000 input variables, so we need the multivariable concept of "gradient" to find the direction of steepest descent in that ultra-high-dimensional space.

$$f(x) \xrightarrow{\text{導數 derivative}} f'(x) \quad \Longrightarrow \quad f(w_1, w_2, \dots, w_{13000}) \xrightarrow{\text{梯度 gradient}} \nabla f$$

## 跟我的關聯 Relevance to Me

梯度下降的邏輯跟最佳化控制（如 PID 調參、機器人路徑規劃）很像——都是在某個誤差函數上反覆修正，只是神經網路的參數空間維度高到人腦無法想像。數位孿生裡的感測器校正問題，本質上也是一個最佳化問題，只是目標函數和參數空間不同。

The logic of gradient descent resembles optimization in control systems (e.g., PID tuning, robot path planning) — both iteratively minimize an error function. The difference is that neural networks operate in a parameter space too high-dimensional for humans to visualize. Sensor calibration in digital twins is fundamentally the same kind of optimization problem, just with a different objective function and parameter space.

## 還沒搞懂的 Still Unclear

- [ ] 反向傳播實際上是怎麼高效計算出 13,000 維梯度的？（下一集）  
      How does backpropagation actually compute the 13,000-dimensional gradient efficiently? (Next episode)
- [ ] 學習率（步長）太大或太小分別會發生什麼？有自動調整的方法嗎？  
      What happens if the learning rate is too large or too small? Are there methods to auto-adjust it?
- [ ] 隨機梯度下降（SGD）跟這裡說的梯度下降有什麼差別？  
      What's the difference between stochastic gradient descent (SGD) and the gradient descent described here?

## 推薦資料 References

- Michael Nielsen - Neural Networks and Deep Learning  
  - http://neuralnetworksanddeeplearning.com/
  - https://github.com/mnielsen/neural-networks-and-deep-learning

## 連結 Links

[[梯度下降 Gradient Descent]] · [[成本函數 Cost Function]] · [[反向傳播 Backpropagation]] · [[激活函數 Activation Function]] · [[卷積神經網路 CNN]]
