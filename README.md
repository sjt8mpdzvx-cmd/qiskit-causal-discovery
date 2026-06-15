# 상관관계를 넘어서: 어디에 개입해야 하는가

관측 데이터에서 인과 구조(DAG)를 자동 발견하고, **결과 변수를 바꾸기 위해 어떤 변수에 개입해야 하는지** 추천하는 Streamlit 앱입니다. 인과 구조 탐색에 Grover 알고리즘을 접목한 양자적 시도를 포함합니다.

## 실행 방법

```bash
cd qiskit-causal-discovery
pip install -r requirements.txt
streamlit run app.py
```

## 주요 기능

- **7개 내장 데이터셋**: Sachs 단백질, Asia 폐질환, Sprinkler, Alarm ICU, Auto MPG, Framingham 심장병, Student 성적
- **CSV 업로드** 및 2~4개 변수 선택
- 모든 유효 DAG 열거 및 **BDeu 점수** 기반 고전 전수조사
- **Grover Oracle/Diffuser** 회로 실행 및 측정 분포 시각화 (Multi-run, Score-weighted 선택)
- 정답 구조 대비 **SHD, Precision, Recall, F1** 비교
- 발견된 DAG 기반 **개입 효과 추정** (backdoor adjustment) 및 coverage 신뢰도 배지 포함 타겟 추천
- 결과 변수 방향 설정: "높일수록 좋음" (MPG, 성적) / "줄일수록 좋음" (심장병, 증상)
- **Groq API** 연동 자연어 해석 (선택 — API 키 없이도 전체 기능 사용 가능)

## 시연 데이터셋

5분 발표 시연은 **Sprinkler weather**를 기본 추천합니다. 구조가 직관적이고 변수 4개라 전수조사, 개입 추천, Grover 회로 실행을 안정적으로 설명하기 좋습니다.

## 앱 구조

| 탭 | 내용 |
|---|---|
| 왜 인과관계인가 | 상관관계 ≠ 인과관계 동기부여, 데이터 소개 |
| 인과 구조 발견 | BDeu 전수조사 결과, 정답 대비 비교, 후보 테이블 |
| **개입 추천** | do-calculus 기반 개입 효과 추정, 추천 타겟 하이라이트 |
| 양자적 접근 | Grover 실행, 측정 분포, 복잡도 비교, 한계와 의의 |
| 종합 분석 | 게이지, 레이더 차트, Key Findings, Score Landscape |

## 한계

현재 구현은 개념증명입니다. 이 앱은 관측 데이터만으로 완전한 인과관계를 증명하는 도구가 아니라, 가능한 DAG를 점수화하고 개입 후보를 비교하는 탐색형 도구입니다.

- **Oracle**: BDeu 점수를 양자 회로 안에서 직접 계산하지 않고, 고전적으로 계산한 상위 후보를 marked state로 인코딩합니다.
- **Grover 해석**: 완전한 양자 우위 입증이 아니라, BDeu 상위 후보에 대한 amplitude amplification 시연입니다.
- **비순환 조건**: Grover는 전체 비트 공간을 탐색하므로 순환 그래프가 측정될 수 있어, 유효 DAG만 후처리로 선택합니다.
- **Markov equivalence**: 관측 데이터만으로는 동일한 조건부 독립 관계를 가진 여러 DAG를 구분할 수 없어, 정답 대비 F1이 낮을 수 있습니다.
- **이산화**: 연속형 변수의 3분위 이산화로 인해 개입 효과는 원래 단위가 아닌 이산화 단위 기준입니다.

## 기술 스택

Qiskit 2.0+, qiskit-aer, numpy, pandas, networkx, matplotlib, streamlit, bnlearn
