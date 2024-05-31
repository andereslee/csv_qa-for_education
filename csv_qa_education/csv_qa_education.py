import json
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

PROMPT_TEMPLATE = """
你是一位数据分析助手，专注与分析学生的成绩，你的回应内容取决于用户的请求内容。

1. 对于文字回答的问题，按照这样的格式回答：
   {"answer": "<你的答案写在这里>"}
例如：
   {"answer": "订单量最高的产品ID是'MNWC3-067'"}

2. 如果用户需要一个表格，按照这样的格式回答：
   {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

3. 如果用户的请求适合返回条形图，按照这样的格式回答：
   {"bar": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}

4. 如果用户的请求适合返回折线图，按照这样的格式回答：
   {"line": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}

5. 如果用户的请求适合返回散点图，按照这样的格式回答：
   {"scatter": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}
注意：我们只支持三种类型的图表："bar", "line" 和 "scatter"。

6. "如果用户提问“第n次和第n-1次考试间各班进步最大的人是谁”，按照这样的逻辑一步一步来回答：
    {"在用户上传的表格中，有一栏是“考试（如：第1次考试、第2次考试。。。。第n-1次考试、第n次考试）”，
    你需要用的列是“姓名”、“考试”、“班级”和“排名”，你需要先借用工具计算每个人被用户指定的两次考试对应排名的差值，差值越大代表进步越大（但差值必须要是正数才有意义，忽略掉负值），
    然后根据班级进行分类，每个班中排名差值最大的为进步最大的学生，最后找出返回每个班排名差值最大的学生。
    返回的表格形式为：第一列是班级，第二列是姓名，第三列是进步名次。每个班应该只返回一个姓名和排名"}
    

请将所有输出作为JSON字符串返回。请注意要将"columns"列表和数据列表中的所有字符串都用双引号包围。
例如：{"columns": ["Products", "Orders"], "data": [["32085Lip", 245], ["76439Eye", 178]]}

你要处理的用户请求如下： 
"""


def dataframe_agent(df, openai_api_key, query): 
    model = ChatOpenAI(model="qwen-long",
                       openai_api_key = openai_api_key, 
                       openai_api_base= "https://dashscope.aliyuncs.com/compatible-mode/v1",
                       temperature=0) # no creativity
    agent = create_pandas_dataframe_agent(llm=model,  # agent执行器
                                          df=df,
                                          agent_executor_kwargs={"handle_parsing_errors": True}, 
                                          verbose=True) 
    prompt = PROMPT_TEMPLATE + query
    response = agent.invoke({"input": prompt})
    response_dict = json.loads(response["output"])
    return response_dict


import streamlit as st
from langchain.memory import ConversationBufferMemory
from pygwalker.api.streamlit import StreamlitRenderer
import pandas as pd


def create_chart(input_data, chart_type):
    df_data = pd.DataFrame(input_data["data"], columns=input_data["columns"])
    df_data.set_index(input_data["columns"][0], inplace=True)
    if chart_type == "bar":
        st.bar_chart(df_data)
    elif chart_type == "line":
        st.line_chart(df_data)
    elif chart_type == "scatter":
        st.scatter_chart(df_data)

def progress_max(df):
    df['进步'] = df[df['考试'] == '第2次']['排名'] - df[df['考试'] == '第1次']['排名']
    progress_max = df[df['考试'] == '第2次'].sort_values('进步', ascending=False).groupby('班级').head(1)[['姓名', '班级', '进步']].reset_index(drop=True)
    return progress_max


st.set_page_config(page_title="成绩智能问答程序", layout="wide")

# Add title to the Streamlit page
st.title("🧠 成绩智能问答程序 💡") 

with st.sidebar:
    openai_api_key = st.text_input("请输入密钥：", type="password")

# Function to cache the uploaded CSV file

def df(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return None


# Upload CSV file before selecting tool
uploaded_file = st.file_uploader("上传CSV格式的文件，让AI帮你来分析吧", type="csv")
select_tool = None

if uploaded_file is not None:
    select_tool = st.selectbox("Select Tool", ["一键总结成绩",
                                               "手动成绩可视化", 
                                               "AI成绩分析"])

if select_tool == "手动成绩可视化":
    st.title("📈 手动成绩可视化")
    if uploaded_file is not None:
        df = df(uploaded_file)
        if df is not None:
            renderer = StreamlitRenderer(df, spec="./gw_config.json", spec_io_mode="rw")
            renderer.explorer()
        else:
            st.write("Failed to load CSV file. Please make sure it is correctly formatted.")
    else:
        st.write("Please upload a CSV file.")

elif select_tool == "AI成绩分析":
    st.title("🔎 AI成绩分析")
    data = uploaded_file
    if data:
        st.session_state["df"] = pd.read_csv(data)
        with st.expander("原始数据"):
            st.dataframe(st.session_state["df"])


    query = st.text_area("请输入你关于以上表格的问题，或数据提取请求，或简单的可视化要求（支持散点图、折线图、条形图）：")
    button = st.button("生成回答")

    if button and not openai_api_key:
            st.info("请输入你的OpenAI API密钥")
    #    if button and not openai_api_base:
    #        st.info("请输入你的OpenAI Base URL")
    # if button and "df" not in st.session_state:
    #    st.info("请先上传数据文件")
    if button and "df" in st.session_state:
        with st.spinner("AI正在思考中，请稍等..."):
            response_dict = dataframe_agent(st.session_state["df"], openai_api_key, query)  # 为 dataframe_agent() 函数传递 df 和 query 参数
            if "answer" in response_dict:
                st.write(response_dict["answer"])
            if "table" in response_dict:
                st.table(pd.DataFrame(response_dict["table"]["data"],
                                      columns=response_dict["table"]["columns"]))
            if "bar" in response_dict:
                create_chart(response_dict["bar"], "bar")
            if "line" in response_dict:
                create_chart(response_dict["line"], "line")
            if "scatter" in response_dict:
                create_chart(response_dict["scatter"], "scatter")
            if "progress_max" in response_dict:
                st.write(response_dict["progress_max"])

elif select_tool == "一键总结成绩":

    st.title("🎯 一键总结成绩")
    button_summarize = st.button("一键总结")
    if button_summarize and uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if df is not None:
            exam_counts = df['考试'].nunique()
            for i in range(1, exam_counts+1):
                exam_data = df[df['考试'] == f'第{i}次']
                exam_summary = exam_data.groupby('班级')['总分'].agg(['mean', 'std']).reset_index()
                st.write(f"第{i}次考试的每个班级总分平均值和标准差:")
                st.table(exam_summary)
            st.write("如果有进一步的需求，欢迎选择“AI数据分析”。")
        else:
            st.write("加载CSV文件失败，请确保格式正确。")
    #else:
        #st.write("请上传一个CSV文件以进行数据总结。")
    st.write("---")
    st.write("Created by 🐯")