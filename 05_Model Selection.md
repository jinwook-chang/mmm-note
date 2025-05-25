
### 5. 모델 구축 및 선택 심층 분석

#### **1. 하이퍼파라미터의 역할 및 종류**

하이퍼파라미터(Hyperparameter)는 모델이 학습을 시작하기 전에 사용자가 설정해야 하는 값들로, 모델의 구조나 학습 방식을 결정합니다. Robyn에서는 이러한 하이퍼파라미터들을 자동으로 탐색하고 최적화하여 모델의 성능을 극대화하려고 시도합니다.

Robyn에서 중요한 하이퍼파라미터들은 주로 다음 두 가지 효과를 모델링하는 데 사용됩니다:

* **광고 이월 효과 (Adstock):** 광고의 효과가 광고 집행 시점 이후에도 일정 기간 지속되는 현상을 모델링합니다.
    * **`thetas` (Geometric Adstock):** 0과 1 사이의 값으로, 광고 효과가 매주 얼마나 남아있는지를 나타냅니다. 값이 클수록 효과가 오래 지속됩니다. (예: theta=0.5이면 매주 50%씩 효과 잔존)
    * **`shapes` 및 `scales` (Weibull Adstock):** 웨이블 분포의 모양(shape)과 크기(scale)를 결정하는 파라미터로, 광고 효과가 시간에 따라 증가했다가 감소하는 패턴(예: 처음에는 효과가 서서히 나타나다 정점을 찍고 감소) 등 더 유연한 형태의 이월 효과를 모델링할 수 있습니다.
        * `shapes`: 광고 효과가 최고점에 도달하는 속도와 형태를 조절합니다. (0 < shape < ∞)
        * `scales`: 광고 효과의 전반적인 지속 기간(길이)을 조절합니다. (0 < scale < 1)

* **반응 곡선 (Saturation / Diminishing Returns):** 광고 지출이 증가함에 따라 반응(예: 매출)의 증가폭이 점차 감소하는 한계효용체감 현상을 모델링합니다.
    * **`alphas` 및 `gammas` (S-shape Curve):** 이 파라미터들은 반응 곡선의 형태를 결정합니다.
        * `alphas`: 곡선의 볼록한 정도(요율)를 제어합니다. (alpha > 0) 값이 작을수록 빨리 포화됩니다.
        * `gammas`: 곡선이 S자 형태를 띠는 정도(변곡점 위치)를 제어합니다. (0 < gamma < 1) 값이 작을수록 낮은 지출 수준에서 변곡점이 나타납니다.

Robyn은 `robyn_inputs()` 함수 내에서 각 채널별로 이러한 하이퍼파라미터들의 탐색 범위(상한값과 하한값)를 설정하도록 합니다. 예를 들어, `facebook_S_alphas = c(0.5, 3)` 와 같이 설정하면 Facebook 채널의 alpha 값을 0.5에서 3 사이에서 탐색합니다.

또한, Robyn은 모델의 과적합(overfitting)을 방지하고 변수 간 다중공선성(multicollinearity) 문제를 완화하기 위해 내부적으로 **릿지 회귀(Ridge Regression)**를 사용합니다. 릿지 회귀의 정규화 강도(lambda) 또한 `Nevergrad`를 통해 자동으로 최적화됩니다.

#### **2. 실험 및 반복 (Iterations & Trials)**

Robyn은 최적의 모델을 찾기 위해 광범위한 탐색 과정을 거칩니다. 이 과정은 `robyn_run()` 함수에서 `iterations`와 `trials`라는 두 가지 주요 매개변수를 통해 제어됩니다.

* **`iterations` (반복 수):** Robyn이 생성하고 평가할 독립적인 모델 후보군의 총 개수를 의미합니다. 각 iteration은 지정된 하이퍼파라미터 범위 내에서 `Nevergrad`에 의해 선택된 특정 하이퍼파라미터 조합으로 하나의 모델을 만듭니다. 예를 들어, `iterations = 2000`으로 설정하면 2000개의 서로 다른 모델 후보군을 생성합니다.
* **`trials` (시도 횟수):** 각 iteration 내에서 `Nevergrad`가 최적의 하이퍼파라미터를 찾기 위해 시도하는 횟수를 의미합니다. `trials` 값이 클수록 `Nevergrad`는 더 넓은 하이퍼파라미터 공간을 탐색하여 해당 iteration에서 더 나은 솔루션을 찾을 가능성이 높아집니다. 예를 들어, `trials = 5`로 설정하면 각 2000개의 iteration마다 5번의 시도를 통해 최적화된 하이퍼파라미터 세트를 찾습니다.

따라서, 총 모델 평가 횟수는 대략 `iterations * trials`가 됩니다. (실제로는 Nevergrad의 작동 방식에 따라 정확히 일치하지 않을 수 있습니다.) 이 값을 너무 작게 설정하면 충분한 탐색이 이루어지지 않아 최적의 모델을 놓칠 수 있고, 너무 크게 설정하면 계산 시간이 매우 길어질 수 있습니다. Robyn 문서에서는 일반적으로 `iterations = 2000`, `trials = 5` (총 10,000개 모델 평가) 정도를 시작점으로 권장하며, 필요에 따라 늘릴 수 있습니다.

#### **3. Pareto Optimal 솔루션 이해**

수천, 수만 개의 모델을 실행하고 나면, 어떤 모델이 "최고"인지 결정해야 합니다. Robyn은 단 하나의 모델만을 선택하는 대신 **Pareto Optimal (파레토 최적) 솔루션**이라는 개념을 사용합니다.

* **Pareto Front (파레토 전선):** 모델의 성능을 나타내는 두 가지 (또는 그 이상)의 상충되는 목표를 동시에 고려했을 때, 어느 한 목표를 개선시키려면 다른 목표가 반드시 나빠지는 지점들의 집합을 의미합니다. Robyn에서는 주로 다음 두 가지를 기준으로 Pareto Front를 구성합니다:
    1.  **모델의 오차 (Model Error):** `NRMSE` (Normalized Root Mean Square Error) 또는 `DECOMP.RSSD` (Decomposition Root Sum of Squared Distance - 실제 값과 모델 예측 값 간의 차이를 나타내는 지표로, 낮을수록 좋음)
    2.  **비즈니스적 합리성/인사이트:** 예를 들어, 광고 채널들의 예산 배분 비율(`total_spend_share`)이 실제 집행된 비율과 얼마나 유사한지, 또는 특정 채널의 ROAS가 비즈니스 상식에 부합하는지 등을 추가로 고려할 수 있습니다. Robyn 1-pager에서는 DECOMP.RSSD를 주로 사용합니다.

```mermaid
 graph LR
     subgraph "Pareto Front 시각화 예시"
         direction LR
         Y["모델 오차 (DECOMP.RSSD) <br> (낮을수록 좋음)"] -- "Y축" --> O(" ");
         X["비즈니스 인사이트 (예: 특정 채널 ROAS) <br> (높거나 특정 범위일수록 좋음)"] -- "X축" --> O;
         O -- " " --> P1["모델 A (오차 낮음, ROAS 보통)"];
         O -- " " --> P2["모델 B (오차 보통, ROAS 좋음)"];
         O -- " " --> P3["모델 C (오차 약간 높음, ROAS 매우 좋음)"];
         style P1 fill:#DCDCDC,stroke:#333,stroke-width:2px
         style P2 fill:#DCDCDC,stroke:#333,stroke-width:2px
         style P3 fill:#DCDCDC,stroke:#333,stroke-width:2px

         classDef pareto fill:#87CEFA,stroke:#0000FF,stroke-width:2px,color:black;
         class P1,P2,P3 pareto;

         S1["열등한 모델 1"] --> P1;
         S2["열등한 모델 2"] --> P2;
         S3["열등한 모델 3"] --> P3;

         note right of P1
          NRMSE: 0.1
          채널 X ROAS: 2.5
         end

          note right of P2
          NRMSE: 0.15
          채널 X ROAS: 3.5
         end

          note right of P3
          NRMSE: 0.2
          채널 X ROAS: 4.0
         end
     end

     P1 -- "선택 후보" --> Z("분석가 최종 판단");
     P2 -- "선택 후보" --> Z;
     P3 -- "선택 후보" --> Z;
```

* **모델 선택 과정:**
    1.  Robyn은 `robyn_outputs()` 함수 실행 후 Pareto Front에 해당하는 모델 ID들과 관련 지표들을 보여줍니다.
    2.  분석가는 이 모델들의 `OnePager` (각 모델의 상세 결과 요약 리포트)를 검토합니다.
    3.  OnePager에는 채널별 기여도, ROAS, 반응 곡선, 이월 효과 등이 시각화되어 있어, 각 모델이 비즈니스 로직에 얼마나 부합하는지, 결과가 안정적인지 등을 판단할 수 있습니다.
    4.  예를 들어, 특정 채널의 ROAS가 비현실적으로 높거나 낮게 나타나는 모델, 또는 특정 채널의 기여도가 예상과 너무 다르게 나오는 모델은 제외할 수 있습니다.
    5.  이러한 과정을 통해 최종적으로 1~3개의 모델을 선택하여 더 깊이 있는 분석이나 예산 최적화 시뮬레이션에 사용합니다.

이처럼 Robyn은 완전 자동화된 단일 최적 모델을 제시하기보다는, 통계적으로 우수하면서도 비즈니스적으로 의미 있는 여러 모델 후보군을 제공하여 분석가의 전문적인 판단을 통해 최종 모델을 선택하도록 유도합니다.

---
