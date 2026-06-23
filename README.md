# 상관관계를 넘어서: 어디에 개입해야 하는가

관측 데이터에서 인과 구조(DAG)를 자동 발견하고, **결과 변수를 바꾸기 위해 어떤 변수에 개입해야 하는지** 추천하는 Streamlit 앱입니다. 인과 구조 탐색에 Grover 알고리즘을 접목한 양자적 시도를 포함합니다.

## 실행 방법

```bash
cd qiskit-causal-discovery
pip install -r requirements.txt
streamlit run app.py
```

회귀 검증은 다음 명령으로 실행합니다.

```bash
python -m unittest discover -s tests -v
```

## 배포 재현성

GitHub Actions는 Python 3.13에서 테스트합니다. Streamlit Community Cloud에 배포할 때도 **Advanced settings → Python version**에서 Python 3.13을 선택하세요. 이미 다른 Python 버전으로 배포된 앱은 Cloud에서 삭제 후 같은 설정으로 다시 배포해야 Python 버전을 변경할 수 있습니다.

## 주요 기능

- **7개 내장 데이터셋**: Sachs 단백질, Asia 폐질환, Sprinkler, Alarm ICU, Auto MPG, Framingham 심장병, Student 성적
- **CSV 업로드** 및 2~8개 변수 선택
- 2~4변수: 모든 유효 DAG 열거 및 **BDeu/BGe 점수** 기반 고전 전수조사
- 5~8변수: 최대 부모 수 제한과 로컬 점수 캐시를 사용한 **hill-climbing 구조 탐색**
- **Grover Oracle/Diffuser** 회로 실행 및 측정 분포 시각화 (Multi-run, Score-weighted 선택)
- 정답 구조 대비 **SHD, Precision, Recall, F1** 비교
- 발견된 DAG 기반 **개입 효과 추정** (backdoor adjustment) 및 coverage 신뢰도 배지 포함 타겟 추천
- 구조 엣지 안정성·개입 효과 95% 구간을 위한 **Bootstrap 신뢰도 분석**
- 회로 자원(깊이, 2큐비트 게이트) 및 선택적 depolarizing **노이즈 시뮬레이션**
- 재현 설정을 포함한 HTML 분석 보고서·개입 효과 CSV 다운로드
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
- **BGe 적용 범위**: BGe는 연속형 수치 변수에만 적용합니다. 이산·범주형 변수는 BDeu를 사용해야 합니다.
- **확장 탐색**: 5~8변수 hill-climbing은 전수조사 결과가 아닌 지역 최적해이며, 양자 시뮬레이션과 직접 비교하지 않습니다.
- **양자 노이즈**: 제공되는 노이즈 모델은 실제 특정 하드웨어의 보정값이 아닌 회로 민감도를 확인하기 위한 단순 depolarizing 모델입니다.

## 기술 스택

Qiskit 2.1+, qiskit-aer, numpy, pandas, networkx, matplotlib, streamlit

`data/download_datasets.py`로 벤치마크 데이터를 다시 내려받으려면 별도로 `bnlearn`을 설치하면 됩니다. 앱 실행에는 필요하지 않습니다.
