
from groq import Groq
import streamlit as st
import os
from crewai import Crew, Agent, Task, Process
from crewai_tools import SerperDevTool
from langchain_groq import ChatGroq



# create title for the streamlit app

st.title('Autonomous Crew Builder')

# create a description for the streamlit app

st.write('This app allows you to create an autonomous crew of agents that can work together to achieve a common goal. You need to define upfront the number of agents that you will use. The agents will work in a sequential order. The agents can be assigned different roles, goals, backstories, tasks and expected outputs. The agents will work together to achieve the common goal. The app will display the output of each agent after the crew has completed its work.')

# ask for the API key in password form
groq_api_key = st.text_input('Enter your GROQ API key', type='password')

serper_api_key = st.text_input('Enter your Serper API key', type='password')


human_input = st.text_area('If you want, you can enter here information that the the agents can use when executing their task')

# ask user in streamlit to enter the number of agents that should be part of the crew

number_of_agents = st.number_input('Enter the number of agents that should be part of the crew', min_value=1, max_value=10, value=1)

namelist = []
rolelist = []
goallist = []
backstorylist = []
taskdescriptionlist = []
outputlist = []
toollist = []

for i in range(0,number_of_agents):
    # ask user in streamlit to enter the name of the agent
    question = 'Enter the name of agent ' + str(i+1)
    agent_name = st.text_input(question)
    namelist.append(agent_name)
    role = st.text_input(f"""Enter the role of agent {agent_name}""" + str(i+1))
    rolelist.append(role)
    goal = st.text_input(f"""Enter the goal of agent {agent_name}""" + str(i+1))
    goallist.append(goal)
    backstory = st.text_input(f"""Describe the backstory of agent {agent_name}""" + str(i+1))
    backstorylist.append(backstory)
    taskdescription = st.text_input(f"""Describe the task of agent {agent_name}""" + str(i+1))
    taskdescriptionlist.append(taskdescription)
    output = st.text_input(f"""Describe the expected output of agent {agent_name}""" +  str(i+1))
    outputlist.append(output)
    search_tool = st.checkbox(f"""Do you want to use the search tool for agent {agent_name}""" + str(i+1))
    if search_tool:
        toollist.append('search')
    else:
        toollist.append('')
    

# create click button 

if st.button('Create Crew'):
    os.environ["GROQ_API_KEY"] = groq_api_key
    os.environ["SERPER_API_KEY"] = serper_api_key
    client = Groq()

    GROQ_LLM = ChatGroq(
            model="llama-3.1-8b-instant"
        )
    search_tool = SerperDevTool()
    agentlist = []
    tasklist = []
    for i in range(0, number_of_agents):
        if toollist[i] == 'search':
            agent = Agent(
                role=rolelist[i],
                goal=goallist[i],
                backstory=backstorylist[i],
                llm=GROQ_LLM,
                verbose=True,
                allow_delegation=False,
                max_iter=5,
                memory=True,
                tool=search_tool
            )
        else:
            agent = Agent(
                role=rolelist[i],
                goal=goallist[i],
                backstory=backstorylist[i],
                llm=GROQ_LLM,
                verbose=True,
                allow_delegation=False,
                max_iter=5,
                memory=True
            )
        agentlist.append(agent)
        task = Task(
            description=taskdescriptionlist[i] + "Consider the input: " + human_input,
            expected_output=outputlist[i],
            agent=agent
        )
        tasklist.append(task)
    crew = Crew(
        agents=agentlist,
        tasks=tasklist,
        process=Process.sequential,
        full_output=True,
        share_crew=False,
    )
    # Kick off the crew's work
    results = crew.kickoff()
    # Print the results
    st.write("Crew Work Results:")
    for i in range(0, number_of_agents):
        st.write(f"Agent {i+1} output: {tasklist[i].output.exported_output}")
else:
    st.write('Please click the button to perform an operation')
