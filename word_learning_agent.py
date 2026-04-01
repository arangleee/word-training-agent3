from __future__ import annotations

import csv
import json
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict


TOPICS = ["일상 대화", "직장", "브랜딩", "사업", "아이디어"]
PRAISE_MESSAGES = [
    "아주 좋습니다. 맥락과 단어 활용이 모두 자연스러워요.",
    "좋은 문장이에요. 실제 대화나 업무 문맥에 바로 써도 어색하지 않습니다.",
    "표현이 안정적이고 전달력이 좋아요. 단어 선택도 정확했습니다.",
    "문장 완성도가 높아요. 단어의 결을 잘 살렸습니다.",
]
MISSION_SCENES: dict[str, list[str]] = {
    "일상 대화": [
        "오랜만에 만난 친구가 요즘 겪는 변화를 털어놓고 있습니다. 분위기는 편하지만 말의 무게는 가볍지 않습니다.",
        "카페에서 둘이 앉아 최근 선택을 돌아보는 중입니다. 한 문장으로 마음을 정리해야 하는 순간입니다.",
        "가족과 저녁 식사 중 대화가 깊어졌습니다. 서로의 입장을 존중하면서도 핵심은 분명히 말해야 합니다.",
    ],
    "직장": [
        "프로젝트 리뷰 회의에서 예상보다 큰 성과가 보고되었습니다. 모두가 그 의미를 정확히 짚는 한마디를 기다립니다.",
        "마감 직전, 팀이 마지막 결정을 내려야 합니다. 판단의 근거를 짧고 선명하게 정리해야 하는 장면입니다.",
        "성과 공유 자리에서 특정 구성원의 기여를 평가해야 합니다. 과장 없이도 실력을 드러내는 문장이 필요합니다.",
    ],
    "브랜딩": [
        "브랜드 리뉴얼 워크숍에서 새 메시지 시안이 공개됐습니다. 방향성이 맞는지 한 문장으로 평가해야 합니다.",
        "캠페인 회고 자리에서 어떤 표현이 고객 반응을 바꿨는지 논의 중입니다. 핵심을 집어내는 문장이 필요합니다.",
        "브랜드 포지셔닝 문구를 확정하기 직전입니다. 팀을 설득할 수 있는 한 문장을 만들어야 합니다.",
    ],
    "사업": [
        "분기 실적 회의에서 숫자 뒤에 있는 흐름을 해석해야 합니다. 리스크와 기회를 함께 담는 문장이 필요합니다.",
        "신규 전략안 발표 직후, 실행 가능성을 평가하는 순서입니다. 논리와 현실감이 동시에 드러나야 합니다.",
        "중요 파트너와 미팅 중 협업의 성과를 정리해야 합니다. 신뢰를 주는 문장 하나가 분위기를 좌우합니다.",
    ],
    "아이디어": [
        "아이디어 피칭에서 예상 밖의 제안이 나왔습니다. 가능성과 한계를 함께 짚는 한 문장이 필요합니다.",
        "화이트보드에 적힌 여러 안 중 하나가 팀의 시선을 붙잡았습니다. 왜 중요한지 문장으로 설명해보세요.",
        "기획 초안 검토 중, 평범한 접근을 뒤집는 발상이 등장했습니다. 그 가치를 표현하는 한마디를 만들어보세요.",
    ],
}


class PromptState(TypedDict, total=False):
    data_dir: str
    available_topics: list[str]
    vocab_rows: list[dict]
    history_rows: list[dict]
    selected_word: dict
    mission_topic: str
    mission_text: str


class EvalState(TypedDict, total=False):
    data_dir: str
    selected_word: dict
    mission_topic: str
    mission_text: str
    user_sentence: str
    pass_status: Literal["pass", "improve"]
    feedback: str
    better_sentence: str
    praise: str
    saved_record: dict


@dataclass
class AgentArtifacts:
    prompt_graph: object
    eval_graph: object


def _clean_text(value: str) -> str:
    return " ".join((value or "").replace("\n", " ").split()).strip()


def _data_paths(data_dir: str) -> dict[str, Path]:
    root = Path(data_dir)
    return {
        "words": root / "words.csv",
        "history": root / "learning_history.csv",
        "profile": root / "user_profile.json",
    }


def ensure_data_files(data_dir: str) -> None:
    paths = _data_paths(data_dir)
    paths["history"].parent.mkdir(parents=True, exist_ok=True)

    needs_history_header = (not paths["history"].exists()) or paths["history"].stat().st_size == 0
    if needs_history_header:
        with paths["history"].open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "word",
                    "topic",
                    "mission",
                    "user_sentence",
                    "pass_status",
                    "feedback",
                    "better_sentence",
                    "praise",
                ],
            )
            writer.writeheader()

    needs_profile_init = (not paths["profile"].exists()) or paths["profile"].stat().st_size == 0
    if needs_profile_init:
        with paths["profile"].open("w", encoding="utf-8") as f:
            json.dump({"sessions": 0, "last_session_at": None}, f, ensure_ascii=False, indent=2)


def load_words(words_csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with words_csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        first_line = f.readline().strip().lower()
        if "daynumber" not in first_line:
            header_line = f.readline()
        else:
            header_line = first_line
        if not header_line:
            return rows
        # Rewind and skip the optional metadata first line.
        f.seek(0)
        first = f.readline()
        if "daynumber" in first.lower():
            f.seek(0)
        reader = csv.DictReader(f)
        for row in reader:
            word = _clean_text(row.get("word", ""))
            meaning = _clean_text(row.get("meaning", ""))
            situation = _clean_text(row.get("situation", ""))
            similar = _clean_text(row.get("similarWord", ""))
            if not word:
                continue
            rows.append(
                {
                    "word": word,
                    "meaning": meaning,
                    "situation": situation,
                    "similar": similar,
                }
            )
    return rows


def load_history(history_csv_path: Path) -> list[dict]:
    if not history_csv_path.exists():
        return []
    with history_csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _select_word_weighted(vocab_rows: list[dict], history_rows: list[dict]) -> dict:
    if not vocab_rows:
        raise ValueError("words.csv에 단어가 없습니다.")

    wrong_count: dict[str, int] = {}
    recent_seen: list[str] = []
    for item in history_rows[-100:]:
        word = item.get("word", "")
        if not word:
            continue
        recent_seen.append(word)
        if item.get("pass_status") == "improve":
            wrong_count[word] = wrong_count.get(word, 0) + 1

    unique_recent = recent_seen[-20:]
    weighted = []
    for row in vocab_rows:
        word = row["word"]
        weight = 1.0
        if word not in unique_recent:
            weight += 1.5
        weight += wrong_count.get(word, 0) * 1.2
        weighted.append((row, weight))

    population = [row for row, _ in weighted]
    weights = [w for _, w in weighted]
    return random.choices(population, weights=weights, k=1)[0]


def _build_mission(word_row: dict, topic: str) -> str:
    word = word_row["word"]
    meaning = word_row.get("meaning", "")
    hint = word_row.get("situation", "")
    scene_candidates = MISSION_SCENES.get(topic) or MISSION_SCENES["일상 대화"]
    scene = random.choice(scene_candidates)
    if hint:
        scene = f"{scene}\n참고 맥락: {hint}"
    return (
        f"[주제: {topic}]\n"
        f"{scene}\n"
        f"이 장면에서 '{word}'를 자연스럽게 넣어 한 문장을 써보세요.\n\n"
        f"(뜻 참고: {meaning})"
    )


def prompt_load_vocab(state: PromptState) -> PromptState:
    paths = _data_paths(state["data_dir"])
    state["vocab_rows"] = load_words(paths["words"])
    return state


def prompt_load_history(state: PromptState) -> PromptState:
    paths = _data_paths(state["data_dir"])
    state["history_rows"] = load_history(paths["history"])
    return state


def prompt_select_word(state: PromptState) -> PromptState:
    state["selected_word"] = _select_word_weighted(
        state.get("vocab_rows", []), state.get("history_rows", [])
    )
    return state


def prompt_generate_mission(state: PromptState) -> PromptState:
    topics = state.get("available_topics") or TOPICS
    topic = random.choice(topics)
    state["mission_topic"] = topic
    state["mission_text"] = _build_mission(state["selected_word"], topic)
    return state


def _evaluate_sentence(state: EvalState) -> tuple[str, str, str, str]:
    sentence = _clean_text(state.get("user_sentence", ""))
    word = state["selected_word"]["word"]
    meaning = state["selected_word"].get("meaning", "")
    similar = state["selected_word"].get("similar", "")
    mission_text = state.get("mission_text", "")
    topic = state.get("mission_topic", "")

    llm_result = _evaluate_with_llm(
        word=word,
        meaning=meaning,
        topic=topic,
        mission_text=mission_text,
        user_sentence=sentence,
    )
    if llm_result is not None:
        pass_status, feedback, better, praise = llm_result
        if similar:
            feedback += f" (비슷한 말: {similar})"
        return pass_status, feedback, better, praise

    used_word = _word_used_in_sentence(word, sentence)
    natural = len(sentence) >= 8 and sentence.endswith((".", "!", "?", "다", "요"))

    if used_word and natural:
        pass_status = "pass"
        feedback = (
            f"좋아요. '{word}'를 문장 안에서 자연스럽게 사용했어요. "
            f"뜻('{meaning}')도 흐름상 잘 맞습니다."
        )
        better = _generate_reference_sentence(word=word, topic=topic, mission_text=mission_text)
        praise = random.choice(PRAISE_MESSAGES)
    elif used_word:
        pass_status = "improve"
        feedback = (
            f"단어 '{word}'를 넣은 점은 좋아요. 다만 문장 마무리나 호흡을 조금 다듬으면 더 자연스러워집니다."
        )
        better = _generate_reference_sentence(word=word, topic=topic, mission_text=mission_text)
        praise = ""
    else:
        pass_status = "improve"
        feedback = (
            f"핵심 단어 '{word}'가 문장에 직접 들어가지 않았어요. "
            "이번 미션에서는 해당 단어를 반드시 포함해 보세요."
        )
        better = _generate_reference_sentence(word=word, topic=topic, mission_text=mission_text)
        praise = ""

    if similar:
        feedback += f" (비슷한 말: {similar})"
    return pass_status, feedback, better, praise


def _evaluate_with_llm(
    *,
    word: str,
    meaning: str,
    topic: str,
    mission_text: str,
    user_sentence: str,
) -> tuple[str, str, str, str] | None:
    """
    Use LLM when OPENAI_API_KEY exists.
    Falls back to rule-based evaluator on any error.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return None

    try:
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        system_prompt = (
            "너는 한국어 어휘 학습 튜터다. 학습자의 문장을 평가해라.\n"
            "반드시 JSON만 출력하고, 키는 pass_status, feedback, better_sentence, praise를 사용해라.\n"
            "규칙:\n"
            "1) pass_status는 pass 또는 improve\n"
            "2) better_sentence는 학습자 문장 교정이 아니라, 미션 상황에 맞는 모범 답안 한 문장\n"
            "3) better_sentence에는 반드시 학습 단어를 포함할 것\n"
            "4) 단어를 쓰지 않았거나 어색하면 improve\n"
            "5) praise는 pass일 때만 1문장, improve면 빈 문자열\n"
            "6) 결과는 한국어로 작성"
        )
        user_prompt = (
            f"[주제]\n{topic}\n\n"
            f"[미션]\n{mission_text}\n\n"
            f"[학습 단어]\n{word}\n"
            f"[뜻]\n{meaning}\n\n"
            f"[학습자 문장]\n{user_sentence}"
        )
        response = model.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        text = _clean_text(response.content if isinstance(response.content, str) else str(response.content))
        # Strip optional markdown fence.
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(text)

        pass_status = data.get("pass_status", "improve")
        if pass_status not in {"pass", "improve"}:
            pass_status = "improve"
        feedback = _clean_text(data.get("feedback", ""))
        better = _clean_text(data.get("better_sentence", ""))
        praise = _clean_text(data.get("praise", ""))

        if not better:
            better = _generate_reference_sentence(word=word, topic=topic, mission_text=mission_text)
        if _clean_text(better) == _clean_text(user_sentence):
            better = _generate_reference_sentence(word=word, topic=topic, mission_text=mission_text)
        if not _word_used_in_sentence(word, better):
            better = _generate_reference_sentence(word=word, topic=topic, mission_text=mission_text)
        if pass_status == "pass" and not praise:
            praise = random.choice(PRAISE_MESSAGES)
        if pass_status == "improve":
            praise = ""

        return pass_status, feedback, better, praise
    except Exception:
        return None


def _generate_reference_sentence(*, word: str, topic: str, mission_text: str) -> str:
    """
    Create a model answer sentence for the presented scene.
    This is independent from the user's input.
    """
    prompt = _clean_text(mission_text)
    if "회의" in prompt or topic == "직장":
        return f"회의에서 나온 제안은 실행 전략까지 갖춘 만큼, 정말 {word} 아이디어라고 평가할 수 있습니다."
    if "브랜드" in prompt or topic == "브랜딩":
        return f"이번 캠페인 문구는 브랜드의 핵심 가치를 간결하게 담아, {word} 완성도를 보여줍니다."
    if "사업" in prompt or "매출" in prompt or topic == "사업":
        return f"이번 전략은 시장 흐름과 실행 우선순위를 함께 짚어낸 점에서 {word} 판단으로 보입니다."
    if "아이디어" in prompt or topic == "아이디어":
        return f"이 제안은 문제 정의부터 실행 그림까지 연결돼 있어 {word} 발상이라고 할 만합니다."
    return f"이 상황을 한 문장으로 정리하면, 지금의 선택은 충분히 {word} 판단이라고 볼 수 있습니다."


def _word_used_in_sentence(word: str, sentence: str) -> bool:
    """
    Accepts exact match + common Korean inflections for predicate endings.
    Examples:
    - 부침이 있다 -> 부침이 있어서 / 부침이 있는
    - 괄목하다 -> 괄목해서 / 괄목하는 / 괄목했다
    """
    w = _clean_text(word)
    s = _clean_text(sentence)
    if not w or not s:
        return False
    if w in s:
        return True

    # Handle predicates ending with 다 (e.g., 하다/있다/되다/없다/맞다...)
    if w.endswith("다") and len(w) >= 2:
        stem = w[:-1]  # keeps predicate stem, e.g. "부침이 있", "괄목하"
        stem_pattern = re.escape(stem)
        if re.search(rf"{stem_pattern}[가-힣]*", s):
            return True
        # 하다 -> 해/한/할/했다 variants (e.g., 괄목하다 -> 괄목할, 괄목해)
        if w.endswith("하다") and len(w) >= 3:
            base = re.escape(w[:-2])  # remove '하다'
            if re.search(rf"{base}(하|해|했|한|할)[가-힣]*", s):
                return True

    # Space-insensitive fallback for multi-word expressions.
    if " " in w:
        compact_w = w.replace(" ", "")
        compact_s = s.replace(" ", "")
        if compact_w in compact_s:
            return True
        if compact_w.endswith("다") and len(compact_w) >= 2:
            stem = re.escape(compact_w[:-1])
            if re.search(rf"{stem}[가-힣]*", compact_s):
                return True

    return False


def _rewrite_sentence_naturally(word: str, sentence: str) -> str:
    """
    Lightweight rewrite for readability without external LLM calls.
    Keeps user's intent but smooths common awkward patterns.
    """
    s = _clean_text(sentence)
    if not s:
        return f"회의 맥락에서 '{word}'를 자연스럽게 활용한 한 문장을 작성해보세요."

    # Common phrasing cleanup.
    replacements = [
        ("저는 오늘 ", "오늘 "),
        ("라는 표현을", "라는 표현을"),
        ("라는 단어를", "라는 단어를"),
        ("해 봤어요", "해봤어요"),
        ("있어서서", "있어서"),
    ]
    for src, dst in replacements:
        s = s.replace(src, dst)
    s = s.replace(f"{word}라는", f"'{word}'라는")

    # Ensure polite and complete ending.
    if not s.endswith((".", "!", "?", "다", "요")):
        s += "."
    if s.endswith("다") and not s.endswith("다."):
        s += "."
    if s.endswith("요") and not s.endswith("요."):
        s += "."

    # If rewrite is still too short, make it more informative.
    if len(s) < 12:
        s = f"해당 맥락에서 '{word}'를 써서 말하면, {s.rstrip('.') }."

    if _clean_text(s) == _clean_text(sentence):
        if word in s:
            s = s.replace(word, f"'{word}'", 1)
        # Keep declarative endings natural (e.g. -습니다 / -다 / -요).
        if not s.endswith((".", "!", "?")):
            s += "."
    return s


def eval_evaluate_answer(state: EvalState) -> EvalState:
    pass_status, feedback, better, praise = _evaluate_sentence(state)
    state["pass_status"] = pass_status
    state["feedback"] = feedback
    state["better_sentence"] = better
    state["praise"] = praise
    return state


def eval_generate_feedback(state: EvalState) -> EvalState:
    return state


def eval_update_progress(state: EvalState) -> EvalState:
    paths = _data_paths(state["data_dir"])
    timestamp = datetime.now(timezone.utc).isoformat()
    record = {
        "timestamp": timestamp,
        "word": state["selected_word"]["word"],
        "topic": state["mission_topic"],
        "mission": state["mission_text"],
        "user_sentence": state["user_sentence"],
        "pass_status": state["pass_status"],
        "feedback": state["feedback"],
        "better_sentence": state["better_sentence"],
        "praise": state["praise"],
    }
    with paths["history"].open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        writer.writerow(record)
    state["saved_record"] = record
    return state


def build_agent() -> AgentArtifacts:
    prompt_builder = StateGraph(PromptState)
    prompt_builder.add_node("load_vocab", prompt_load_vocab)
    prompt_builder.add_node("load_user_history", prompt_load_history)
    prompt_builder.add_node("select_word", prompt_select_word)
    prompt_builder.add_node("generate_mission", prompt_generate_mission)
    prompt_builder.add_edge(START, "load_vocab")
    prompt_builder.add_edge("load_vocab", "load_user_history")
    prompt_builder.add_edge("load_user_history", "select_word")
    prompt_builder.add_edge("select_word", "generate_mission")
    prompt_builder.add_edge("generate_mission", END)

    eval_builder = StateGraph(EvalState)
    eval_builder.add_node("evaluate_answer", eval_evaluate_answer)
    eval_builder.add_node("generate_feedback", eval_generate_feedback)
    eval_builder.add_node("update_progress", eval_update_progress)
    eval_builder.add_edge(START, "evaluate_answer")
    eval_builder.add_edge("evaluate_answer", "generate_feedback")
    eval_builder.add_edge("generate_feedback", "update_progress")
    eval_builder.add_edge("update_progress", END)

    return AgentArtifacts(
        prompt_graph=prompt_builder.compile(),
        eval_graph=eval_builder.compile(),
    )
