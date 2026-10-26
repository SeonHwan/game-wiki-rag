from pydantic import BaseModel


class EvalCase(BaseModel):
    question: str
    expected_source: str   # 정답이 들어있는 파일명 (예: "patch_notes.md")
    reference: str         # 정답 텍스트 (ragas의 context_precision/recall이 요구)


EVAL_CASES: list[EvalCase] = [
    # patch_notes.md (3) — "OO가 패치로 바뀌었어" 패턴 반복분 제거, 의미 있는 케이스만 유지
    EvalCase(
        question="화염룡은 이번 패치에서 어떻게 바뀌었어?",
        expected_source="patch_notes.md",
        reference="화염룡의 화염 브레스 공격력이 20% 감소(너프)했습니다.",
    ),
    EvalCase(
        question="얼음 골렘 이동속도가 왜 빨라졌어?",
        expected_source="patch_notes.md",
        reference="너무 느려서 회피가 쉽다는 의견을 반영해 이동 속도를 15% 증가시켰습니다.",
    ),  # 리랭커로 해결한 실패 사례
    EvalCase(
        question="도적 길드 평판이 낮아도 상점 쓸 수 있어?",
        expected_source="patch_notes.md",
        reference="네, 평판이 음수여도 일부 상점을 이용할 수 있도록 변경되었습니다.",
    ),  # 과거 오검색 사례

    # bestiary.md (4) — 약점/드롭/위치/주의사항 유형별 1개씩만
    EvalCase(
        question="화염룡의 약점은 뭐야?",
        expected_source="bestiary.md",
        reference="꼬리가 약점이며, 빙결 상태이상일 때 추가 피해를 입어 얼음 속성 무기가 유리합니다.",
    ),
    EvalCase(
        question="슬라임 잡으면 뭐 나와?",
        expected_source="bestiary.md",
        reference="소량의 골드와 점액 재료를 드롭합니다.",
    ),
    EvalCase(
        question="고블린은 어디서 나와?",
        expected_source="bestiary.md",
        reference="숲 지역에 무리 지어 서식합니다.",
    ),
    EvalCase(
        question="트롤을 상대할 때 뭘 조심해야 해?",
        expected_source="bestiary.md",
        reference="체력이 높고 재생 능력이 있어, 화염 속성 공격으로 재생을 억제하지 않으면 장기전이 됩니다.",
    ),

    # town_news.md (2) — 대표 2개만
    EvalCase(
        question="여관에서 뭐 새로 팔아?",
        expected_source="town_news.md",
        reference="마를렌 여관에서 감자 수프를 새 메뉴로 판매합니다.",
    ),
    EvalCase(
        question="무기 강화 할인하는 곳 있어?",
        expected_source="town_news.md",
        reference="대장장이 고든이 한정 기간 무기 강화 비용을 30% 할인합니다 (전설 등급 제외).",
    ),

    # quests.md (3) — "보상이 뭐야" 중복(고블린) 제거
    EvalCase(
        question="트롤 잡는 퀘스트 보상이 뭐야?",
        expected_source="quests.md",
        reference="'다리 밑 괴물' 퀘스트로, 트롤을 처치하면 금화 20개와 트롤 가죽 방어구를 받습니다.",
    ),
    EvalCase(
        question="화염룡 잡는 퀘스트는 누가 주는 거야?",
        expected_source="quests.md",
        reference="왕국 기사단이 의뢰하는 '용의 둥지' 퀘스트입니다.",
    ),
    EvalCase(
        question="사제 엘리안이 주는 퀘스트는 뭐야?",
        expected_source="quests.md",
        reference="'잊혀진 자들의 안식' 퀘스트로, 스켈레톤 워리어 5마리를 정화하면 완료됩니다.",
    ),
]
