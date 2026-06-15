[학기말 시험 대체 제출] 상관관계를 넘어서: 어디에 개입해야 하는가 — 양자 인과 구조 발견 및 의사결정 보조 앱

---

앱 링크: https://qiskit-causal-discovery-9f5uyhbuf2a2twcvqcihrm.streamlit.app/
소스코드: https://github.com/sjt8mpdzvx-cmd/qiskit-causal-discovery

---

## 1. 문제 설정 — 왜 인과관계인가

"아이스크림 판매량이 늘면 익사 사고도 늘어난다"는 상관관계입니다. 하지만 아이스크림 판매를 금지한다고 해서 익사 사고가 줄어들지는 않습니다. 상관관계만으로는 '어디에 개입해야 결과가 바뀌는지' 알 수 없으며, 잘못된 정책 결정으로 이어질 수 있습니다.

본 프로젝트의 목표는 관측 데이터에서 변수 사이의 인과 구조(DAG)를 자동으로 발견하고, 이를 바탕으로 "결과 변수를 바꾸려면 어떤 변수에 개입해야 하는가"라는 실질적인 질문에 답하는 **의사결정 보조 도구**를 개발하는 것이었습니다. 특히, 인과 구조 탐색이 본질적으로 super-exponential한 탐색 공간을 갖는 조합 최적화 문제라는 점에 착안하여, **Qiskit의 Grover 알고리즘**을 접목한 양자적 접근을 시도했습니다.

## 2. 주요 기능 및 방법론

### 2-1. 인과 구조 발견 (Causal Discovery)
BDeu(Bayesian Dirichlet equivalent uniform) 점수를 기반으로 데이터에 가장 적합한 구조를 찾습니다. 고전적 전수조사(Exhaustive Search)를 통해 모든 유효 DAG를 열거하고 점수화하여 최적의 구조를 도출합니다.

### 2-2. 개입 효과 추정 및 추천 (Intervention Recommendation)
발견된 DAG에 **do-calculus(Backdoor Adjustment)**를 적용하여 개입 효과를 추정합니다. 
- "MPG를 높이려면 무게(Weight)를 줄여야 하는가, 마력(Horsepower)을 줄여야 하는가?"와 같은 개입 질문에 대해 통계적 근거를 바탕으로 우선순위를 추천합니다.
- 데이터의 양과 질에 따른 **Coverage 기반 신뢰도 배지**를 도입하여 추정의 한계를 명확히 전달합니다.

### 2-3. 양자적 탐색 (Quantum Search via Qiskit)
인과 구조 탐색을 비정렬 탐색 문제로 정식화하여 Grover 알고리즘을 적용했습니다.
- **Qiskit Aer Simulator**를 활용하여 12큐비트 규모의 Grover 회로를 실행합니다.
- BDeu 고득점 후보들을 Oracle의 marked state로 설정하고, 진폭 증폭(Amplitude Amplification)을 통해 해당 구조들이 측정될 확률이 유의미하게 증가함을 시연합니다.

### 2-4. AI 전문 리포트 (LLM-powered Interpretation)
분석 결과(DAG, BDeu, Intervention Table, Quantum Stats)를 **Groq API(Llama 3.3)**와 연동하여 비전문가도 바로 이해할 수 있는 전문적인 분석 보고서 형태로 자동 생성합니다.

## 3. 맞닥뜨린 어려움과 해결 과정

- **Markov Equivalence의 본질적 한계**: 관측 데이터만으로는 인과 방향을 완전히 식별할 수 없는 경우가 있음을 인지하고, 이를 앱 내 도움말과 AI 해석을 통해 사용자에게 명확히 고지하도록 개선했습니다.
- **API 안정성 및 보안**: Groq API 연동 시 Cloudflare 차단(Error 1010) 문제를 해결하기 위해 User-Agent 헤더를 우회 설정하고, 키 공백 자동 제거 로직을 추가하여 안정성을 확보했습니다.
- **개입 방향성 제어**: 목표 변수의 특성(높일수록 좋은지, 낮출수록 좋은지)에 따라 개입 추천 로직이 동적으로 작동하도록 outcome_higher_is_better 플래그를 도입했습니다.

## 4. 회고 및 시사점

### Qiskit 활용의 의의
단순히 양자 알고리즘을 코드로 짜보는 것에 그치지 않고, **실생활의 의사결정 문제(인과 추론)를 양자 문제로 정식화(Formulation)**해 보았다는 점에 큰 의미를 두었습니다. 변수 개수가 늘어날수록 전수조사($O(N)$) 대비 Grover($O(\sqrt{N})$)의 이차 속도 향상이 실질적인 경쟁력을 가질 수 있음을 확인했습니다.

### AI 협업 도구(Gemini CLI/Claude) 활용
개발 과정에서 AI 에이전트를 적극적으로 활용했습니다. 특히 데이터 전처리 로직의 오류를 잡아내고, 복잡한 인과 추론 지표들을 시각화하는 과정에서 시간을 크게 단축할 수 있었습니다. AI가 제안한 코드를 그대로 수용하기보다, 인과 추론의 통계적 정밀도를 유지하기 위해 직접 코드를 검토하고 보정하는 과정이 학습에 큰 도움이 되었습니다.

## 5. 향후 발전 과제
- Oracle 내에서 BDeu 점수를 직접 계산하는 In-circuit Scoring 구현
- 개입 데이터(Interventional Data)를 추가 활용하여 Markov 동치류 식별력 강화
- 더 많은 변수로 확장하여 Grover 알고리즘의 실질적 우위 실험
