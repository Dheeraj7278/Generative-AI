import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler


##set up upi the streamlit app
st.set_page_config(page_title="Text to Math Problem Solver and Data search Assistant",page_icon="emoji")
st.title("Text to Math Problem Solver Using Google Gemma 2")

groq_api_key=st.sidebar.text_input(label="Groq Api Key",type="password")


if not groq_api_key:
    st.info("Please add your Groq API Key to continue.")
    st.stop()

llm=ChatGroq(model="llama3-8b-8192",groq_api_key=groq_api_key)

##Initializing the tools
wikipedia_wrapper=WikipediaAPIWrapper()

wikipedia_tool=Tool(
    name="wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the Internet to find the various information on the "
)

#Initialize the math tool
math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tools for answering math questions.only input mathematical expression needs to be provided"
)

prompt="""
your agent tasked for solving users mathematical question.Logically arrive at the solution and provide the detailed explaination
and display it point wise for the question below.
Question:{question}
Answer:
"""
prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

## combine all the tools into chain
chain=LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool=Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)


#initialize the agents
assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages"  not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi i'm a Math chatbot who can answer your all maths questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

#function to generate  the response 
def generate_response(question):
    response=assistant_agent.invoke({'input':question})
    return response


#Lets start the interaction
question =st.text_area("Enter your question:","A factory produces 1200 widgets in 8 hours with 10 workers. How many widgets can 15 workers produce in 6 hours, assuming the same rate of productivity?")

if st.button("find my answer"):
    if question:
        with st.spinner("Generate response..."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb])

            st.session_state.messages.append({'role':'assistant','content':response})
            st.write("###Response:")
            st.success(response)
    else:
        st.warning("please enter the question")


