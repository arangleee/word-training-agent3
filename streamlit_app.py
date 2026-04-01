from __future__ import annotations

from pathlib import Path

import streamlit as st

from word_learning_agent import TOPICS, build_agent, ensure_data_files


st.set_page_config(page_title="Word Training Agent", page_icon="📘", layout="centered")
st.title("📘 우리말 단어 학습 에이전트")

DATA_DIR = Path(__file__).parent / "data"
ensure_data_files(str(DATA_DIR))

if "agent" not in st.session_state:
    st.session_state.agent = build_agent()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_round" not in st.session_state:
    st.session_state.current_round = None
if "today_count" not in st.session_state:
    st.session_state.today_count = 0
if "session_ended" not in st.session_state:
    st.session_state.session_ended = False

with st.sidebar:
    st.markdown("### 설정")
    st.caption("주제는 매 라운드 랜덤으로 선택됩니다.")
    st.write("가능한 주제:", ", ".join(TOPICS))
    st.write(f"오늘 학습한 단어 수: {st.session_state.today_count}")


def start_new_round() -> None:
    result = st.session_state.agent.prompt_graph.invoke(
        {
            "data_dir": str(DATA_DIR),
            "available_topics": TOPICS,
        }
    )
    st.session_state.current_round = {
        "selected_word": result["selected_word"],
        "mission_topic": result["mission_topic"],
        "mission_text": result["mission_text"],
    }
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": (
                "오늘의 단어를 골랐어요.\n\n"
                f"**단어:** {result['selected_word']['word']}\n\n"
                f"{result['mission_text']}"
            ),
        }
    )


if st.session_state.current_round is None:
    st.info("학습 시작을 누르면 단어를 하나 선택해 미션을 드려요.")
    if st.button("학습 시작", type="primary"):
        st.session_state.session_ended = False
        start_new_round()
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.current_round is not None:
    user_input = st.chat_input("미션에 맞는 한 문장을 작성해보세요.")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        out = st.session_state.agent.eval_graph.invoke(
            {
                "data_dir": str(DATA_DIR),
                "selected_word": st.session_state.current_round["selected_word"],
                "mission_topic": st.session_state.current_round["mission_topic"],
                "mission_text": st.session_state.current_round["mission_text"],
                "user_sentence": user_input,
            }
        )
        response = (
            f"**평가 결과:** `{out['pass_status']}`\n\n"
            f"**피드백:** {out['feedback']}\n\n"
            f"**더 자연스러운 표현 예시:** {out['better_sentence']}"
        )
        if out.get("praise"):
            response += f"\n\n🎉 {out['praise']}"
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.today_count += 1
        st.session_state.current_round = None
        st.rerun()

if st.session_state.current_round is None and st.session_state.today_count > 0:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("한 단어 더", use_container_width=True):
            st.session_state.session_ended = False
            start_new_round()
            st.rerun()
    with col2:
        if st.button("오늘은 여기까지", use_container_width=True):
            st.session_state.session_ended = True
            st.session_state.current_round = None
            st.rerun()
        st.caption("채팅 기록은 유지되고, 결과는 data/learning_history.csv에 저장됩니다.")

if st.session_state.session_ended:
    st.success("오늘 학습을 마쳤어요. 수고했어요! 내일 또 이어서 해봐요.")
