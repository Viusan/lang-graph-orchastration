import os
import sqlite3
import io
import contextlib

from dotenv import load_dotenv
from openai import OpenAI
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

load_dotenv()

api_key = os.getenv("OSBOT_API_KEY")
base_url = os.getenv("OSBOT_BASE_URL")
model_name = os.getenv("LOCAL_MODEL")

client = OpenAI(
    api_key="dummy",
    base_url=base_url,
    default_headers={"x-api-key": api_key},
)

#this is NOT an agent (this is our state), it's a data container (or shared memory) that gets passed around the agents.
class ResearchState(TypedDict):
    given_data: str
    calculated_data: dict
    interpretation: str
    reviewer_feedback: str
    reviewer_verdict: bool
    revision_counter: int
    decision_log: list[dict]

#every LangGraph node is just a function with this shape
def data_agent(state: ResearchState) -> dict:
    conn = sqlite3.connect("diabetes.db")

    query = """
    SELECT PhysActivity, Diabetes_binary, COUNT(*) as antall
    FROM diabetes_binary
    GROUP BY PhysActivity, Diabetes_binary
    """
    results = conn.execute(query).fetchall()
    conn.close()

    #create a text summary of the numbers
    text_data = "PhysActivity, Diabetes_binary, Antall\n"
    for row in results:
        text_data += f"{row[0]}, {row[1]}, {row[2]}\n"

    #we create a log so we know what has happened, and append it to decision log in our typeddict
    new_log_entry = {"agent": "data_agent", "action": "generated data from diabetes.db"}
    updated_log = state["decision_log"] + [new_log_entry] #we add list + list, or error. Also in LangGraph creating a new list instead of mutating old is better.

    #we return the stuff we want to update in our state
    return {"given_data": text_data, "decision_log": updated_log}



def run_sandbox(code: str) -> dict: #utility function
    buffer = io.StringIO() #empty fake in-memory
    try:
        with contextlib.redirect_stdout(buffer): #whatever is printed inside here goes into the memory
            exec(code)
            captured_text = buffer.getvalue()   #pulls everything from buffer to string
            return {"success": True, "stdout": captured_text, "error": None}
    except Exception as e:
        return {"success": False, "stdout": buffer.getvalue(), "error": str(e)}

def modelling_agent(state: ResearchState) -> dict:
    conn = sqlite3.connect("diabetes.db")

    query = """
    SELECT PhysActivity, Diabetes_binary, COUNT(*) as antall
    FROM diabetes_binary
    GROUP BY PhysActivity, Diabetes_binary
    """
    results = conn.execute(query).fetchall()
    conn.close()

    count_00 = None #no phys and no diabetes
    count_01 = None #no phys but diabetes
    count_10 = None #phys but no diabetes
    count_11 = None #both phys and diabetes

    for row in results: #asign count to who has what
        phys_activity = row[0]
        diabetes = row[1]
        count = row[2]

        if phys_activity == 0.0 and diabetes == 0.0:
            count_00 = count
        elif phys_activity == 0.0 and diabetes == 1.0:
            count_01 = count
        elif phys_activity == 1.0 and diabetes == 0.0:
            count_10 = count
        elif phys_activity == 1.0 and diabetes == 1.0:
            count_11 = count

    code = f"""
from scipy.stats import chi2_contingency
table = [[{count_00}, {count_01}], [{count_10}, {count_11}]]
chi2, p_value, dof, expected = chi2_contingency(table)
print(chi2, p_value)
"""
    sandbox_result = run_sandbox(code)

    if sandbox_result["success"]:
        chi2_value, p_value = sandbox_result["stdout"].split()
        calculated = {"chi2": float(chi2_value), "p_value": float(p_value)}
    else:
        calculated = {"error": sandbox_result["error"]}

    new_log_entry =  {"agent": "modelling_agent", "action": "sending data and code to run in sandbox"}
    updated_log = state["decision_log"] + [new_log_entry]

    return {"calculated_data": calculated, "decision_log": updated_log}

#new node/agent
def interpretation_agent(state: ResearchState) -> dict:
    calculated= state["calculated_data"]   #read what modelling_agent produced, and what is in state
    
    prompt = f"""Here is a statistical result from a health survey. A chi-square test of independence was run to check whether physical activity (PhysActivity) is related to diabetes (Diabetes_binary) in this data.

Chi-square statistic: {calculated['chi2']}
P-value: {calculated['p_value']}

Based on this result, what can we say about the relationship between physical activity and diabetes? Keep in mind this dataset has a very large sample size, so even a small, practically unimportant difference can produce a statistically significant p-value — be careful not to overstate the finding. Answer briefly, in a maximum of 3 sentences."""

    response = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=1500,
    )
    
    response_text = response.choices[0].message.content

    #update state
    new_log_entry = {"agent": "interpretation_agent", "action": "interpreting the data that was given from data_agent"}
    updated_log = state["decision_log"] + [new_log_entry]

    return {"interpretation": response_text, "decision_log": updated_log}

def reviewer_agent(state: ResearchState) -> dict:
    interpreted_text = state["interpretation"]

    prompt = f"""You are an interpretor agent, you have to review the text that is given to you and come to a conlusion wether the results are reasonable or not.
    Once you are done interpreting the text, end the response with one line and nothing else after is:
    Either "VERDICT: APPROVED" if you agree with the interpretaiob or "VERDICT: REVISE" if you disagree.
    The interpreted text you are going to review is {interpreted_text}
    """
    response = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=1500,
    )

    response_text = response.choices[0].message.content

    if "VERDICT: APPROVED" in response_text.upper():
        reviewer_verdict = True
    else:
        reviewer_verdict = False

    new_log_entry = {"agent": "reviewer_agent", "action": f"deciding if the interpretation was valid or not, this time the verdict was {reviewer_verdict}"}
    updated_log = state["decision_log"] + [new_log_entry]

    return {"reviewer_feedback": response_text, "reviewer_verdict": reviewer_verdict, "revision_counter": state["revision_counter"]+1, "decision_log": updated_log}
    
def route_after_review(state: ResearchState) -> str:
    if state["revision_counter"] < 3: #just so we dont create a infinite loop
        if state["reviewer_verdict"]:
            return "approved"
        else:
            return "needs_revision"
    else:
        return "approved"

#create a mini graph using LangGraph
builder = StateGraph(ResearchState) #we tie it to our state schema

#register the nodes with name asigned
builder.add_node("data_agent", data_agent)
builder.add_node("modelling_agent", modelling_agent)
builder.add_node("interpretation_agent", interpretation_agent)
builder.add_node("reviewer_agent", reviewer_agent)

#wire the steps
builder.add_edge(START, "data_agent")          
builder.add_edge("data_agent", "modelling_agent")  
builder.add_edge("modelling_agent", "interpretation_agent")       
builder.add_edge("interpretation_agent", "reviewer_agent")

#this conditional node is so that if we get false from reviewer, we can loop back until its satisfied
builder.add_conditional_edges(
    "reviewer_agent", route_after_review,
    {
        #if route_after_review function returns approved we end, or we loop back into interpretation
        "approved": END,
        "needs_revision": "interpretation_agent"
    },
)

#compile to something runnable
graph = builder.compile()

#graph.get_graph().draw_mermaid_png(output_file_path="graph.png") #create a graph to vizualise how the nodes interact with each other

result = graph.invoke({
    "given_data": "",
    "calculated_data": {},
    "interpretation": "",
    "reviewer_feedback": "",
    "reviewer_verdict": False,
    "revision_counter": 0,
    "decision_log": [],
})

print(result["interpretation"])
print(result["given_data"])
print(result["decision_log"])
print(result["calculated_data"])