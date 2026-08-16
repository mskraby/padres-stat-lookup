import os
import sqlite3
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun

from fast_path import try_fast_path

KOFI_URL = "https://ko-fi.com/skraby"
FREE_QUESTIONS_PER_SESSION = 10

# ==========================================
# 0. FAILSAFE SESSION INITIALIZATION
# ==========================================
if "langchain_messages" not in st.session_state:
    st.session_state["langchain_messages"] = []
if "question_count" not in st.session_state:
    st.session_state["question_count"] = 0

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def extract_clean_text(raw_response):
    """Extracts clean text from raw dictionary/list/message outputs."""
    if isinstance(raw_response, dict):
        output = raw_response.get("output", "")
    else:
        output = raw_response

    if isinstance(output, list):
        text_parts = []
        for item in output:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
            elif hasattr(item, "content"):
                text_parts.append(extract_clean_text(item.content))
            else:
                text_parts.append(str(item))
        return "\n".join(text_parts) if text_parts else str(output)

    # AIMessage.content can itself be a list-of-parts (Gemini's format) --
    # recurse instead of stringifying it raw, or a plain string comes back as-is.
    if hasattr(output, "content"):
        return extract_clean_text(output.content)

    return str(output)


def synthesize_from_steps(llm, question, intermediate_steps):
    """Fallback for when the agent exhausts max_iterations without emitting a
    Final Answer. `early_stopping_method="generate"` isn't supported for
    tool-calling agents in this langchain version, so this reimplements that
    behavior manually: feed everything the agent already looked up back to
    the LLM for one direct synthesis call, instead of showing the user the
    unhelpful canned "Agent stopped due to max iterations." message.
    """
    steps_text = "\n\n".join(
        f"Query: {action.tool_input}\nResult: {observation}"
        for action, observation in intermediate_steps
        if str(action.tool_input).strip()
    )
    if not steps_text:
        return "I wasn't able to find an answer to that question -- try rephrasing or asking about one thing at a time."

    prompt = f"""Based on these database query results, answer the user's question about San Diego Padres history as directly as possible.
If some part of the question isn't covered by the results, say what's missing rather than guessing.

Question: {question}

Query results:
{steps_text}

Answer:"""
    return extract_clean_text(llm.invoke(prompt))


# ==========================================
# 2. AGENT & DATABASE SETUP (HYBRID SETUP)
# ==========================================
@st.cache_resource
def get_sql_agent():
    api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
    if not api_key:
        st.error("⚠️ Missing GOOGLE_API_KEY! Set it in .streamlit/secrets.toml or environment variables.")
        st.stop()

    os.environ["GOOGLE_API_KEY"] = api_key

    # Connect to SQLite
    db_path = "padres_history.db"
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

    # Initialize Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0
    )

    # Initialize Web Search Tool
    search_tool = DuckDuckGoSearchRun()

    # System instructions preventing infinite loops
    custom_prefix = """
    You are an expert San Diego Padres statistician.
    You have access to a local SQLite database (`padres_history.db`) and a Web Search tool.

    DATABASE GUIDE -- prefer the precomputed tables below over aggregating the raw game logs yourself:
    - padres_career_batting / padres_career_pitching: one row per player, career totals + AVG/ERA. Use for "career leader" / "most X all-time" questions.
    - padres_season_batting / padres_season_pitching: one row per player per season, season totals + AVG/ERA. Use for "in [year]" questions.
    - padres_season_standings: one row per season with Wins/Losses/WinPct (regular season only). Use for "what was their record in [year]" questions.
    - padres_postseason_games: one row per playoff/LDS/LCS/WS game (Date, Opponent, GameType, Win/Loss). Use for postseason/World Series questions.
    - padres_games: one row per every Padres game ever played (regular + postseason), with result and opponent.
    - padres_players: one row per player with Bats/Throws, primary position, and first/last season with the Padres. Use for bio/tenure/position questions.
    - padres_batting_logs / padres_pitching_logs: raw game-by-game box scores. Only query these directly for single-game or date-specific questions (e.g. "how many strikeouts did X have on [date]") that the aggregate tables above can't answer.
    Note: a few players share a name (e.g. two different "Dave Roberts"); those are disambiguated in player_name with a "(years)" suffix -- if a plain name search returns multiple distinct player_name values, ask which one or state both.

    RULES:
    1. Pick the most specific table above for the question, and filter by player_name/Season/Date immediately -- never scan whole tables.
    2. If the local database returns 0 rows or is missing the player/stat, run a single Web Search.
    3. Run AT MOST 2-3 queries per distinct thing the user asked about (a question with two parts, e.g. "record AND who they lost to", may need 2-3 queries for each part). As soon as you have a usable result for a part, move on -- do not run extra confirmation or exploratory queries once you have the answer.
    4. State where your answer came from (e.g., "According to database records..." or "According to MLB history search...").
    """

    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        extra_tools=[search_tool],
        agent_type="tool-calling",
        prefix=custom_prefix,
        max_iterations=8,
        early_stopping_method="force",  # "generate" isn't supported for tool-calling agents in this langchain version; handled manually in synthesize_from_steps()
        verbose=False,
        handle_parsing_errors=True,
        agent_executor_kwargs={"return_intermediate_steps": True},
    )

    return agent_executor, llm


# ==========================================
# 3. STREAMLIT UI & CHAT LOOP
# ==========================================
st.set_page_config(page_title="Padres Stat Lookup", page_icon="⚾")
st.title("⚾ San Diego Padres Stat Lookup")
st.caption("An unofficial fan project. Not affiliated with or endorsed by MLB or the San Diego Padres.")

remaining = max(0, FREE_QUESTIONS_PER_SESSION - st.session_state.question_count)
st.caption(
    f"☕ {remaining} free question{'s' if remaining != 1 else ''} left this session -- "
    f"[support the project on Ko-fi]({KOFI_URL}) if you're enjoying it."
)

agent, llm = get_sql_agent()

# Display chat history
for msg in st.session_state.langchain_messages:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.write(msg["content"])

# User input box
if prompt := st.chat_input("Ask a question about Padres history..."):
    # Render user prompt
    st.session_state.langchain_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Process and render response
    with st.chat_message("assistant"):
        # Try the fast path first: pattern-matched common questions get an instant,
        # zero-Gemini-cost answer straight from SQL. Only questions that don't match
        # a known shape fall through to the full LLM agent below, so the free-question
        # quota only ever gets spent on the ones that actually need the LLM.
        fast_answer = try_fast_path(prompt)
        if fast_answer is not None:
            clean_text = fast_answer
            st.write(clean_text)
        elif st.session_state.question_count >= FREE_QUESTIONS_PER_SESSION:
            clean_text = (
                f"You've used all {FREE_QUESTIONS_PER_SESSION} free questions for this session! "
                f"If you're enjoying the Padres Stat Lookup, consider [supporting it on Ko-fi]({KOFI_URL}) "
                "to help keep it running -- or just refresh the page for another batch of free questions."
            )
            st.write(clean_text)
        else:
            with st.spinner("Searching records..."):
                raw_res = agent.invoke({"input": prompt})
                clean_text = extract_clean_text(raw_res)
                if clean_text.strip() == "Agent stopped due to max iterations." and raw_res.get("intermediate_steps"):
                    clean_text = synthesize_from_steps(llm, prompt, raw_res["intermediate_steps"])
                st.write(clean_text)
            st.session_state.question_count += 1

    st.session_state.langchain_messages.append({"role": "assistant", "content": clean_text})