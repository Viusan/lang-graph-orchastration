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
    method_choice: str
    generated_code: str
    calculated_data: dict
    interpretation: str
    reviewer_feedback: str
    reviewer_verdict: bool
    revision_counter: int
    modelling_counter: int
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

def methods_agent(state: ResearchState) -> dict:
    data = state["given_data"]

    prompt = f"""You are a biostatistics assistant helping choose an appropriate statistical test.

Here is aggregated survey data showing the number of respondents (Antall) for each
combination of two binary variables:

- PhysActivity: whether the respondent reported physical activity in the past 30 days
  (0 = no, 1 = yes)
- Diabetes_binary: whether the respondent has diabetes or prediabetes
  (0 = no, 1 = yes)

here is the data that you are given {data}

Based on this data, decide which statistical method is most appropriate to test whether
there is a significant association between PhysActivity and Diabetes_binary. Briefly
explain your reasoning (e.g. variable types, sample size, what the test assumes), then
end your answer with exactly one line in this format:

METHOD: <name of the test>
"""
    response = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=1500,
    )

    response_text = response.choices[0].message.content

    new_log_entry = {"agent": "methods_agent", "action": "agent decided what method is best to use for calculating the statistics"}
    updated_log = state["decision_log"] + [new_log_entry]

    return {"method_choice": response_text, "decision_log": updated_log}

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
    use_method = state["method_choice"]
    given_data = state["given_data"]
    if "error" in state["calculated_data"]:
        error_note = f"\nYour previous attempt failed with this error: {state['calculated_data']['error']}\nFix the code so this doesn't happen again.\nThe generated code that previously did not work was: {state['generated_code']}\nUsing this information generate new code that works."
    else:
        error_note = ""

    prompt = f"""
    You are a Python data analyst. Write Python code that performs the following statistical
test on the data given below. Do not explain anything — output ONLY valid Python code,
with no markdown formatting, no triple backticks, and no comments outside the code.

Method to use: {use_method}

The given data is: {given_data}

Requirements:
- Use only these libraries: scipy.stats, numpy. Do not import anything else, and do not
  attempt any file access, network access, or database connection — work only with the
  numbers given above.
- Build the data directly from the literal values shown (e.g. as a list or array), not by
  reading any external source.
- Run the test named above and print the result in exactly this format, with no other
  output:
  RESULT: statistic=<value>, p_value=<value>
- The code must run standalone from top to bottom with no undefined variables.
{error_note}
"""

    response = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=1500,
    )

    response_text = response.choices[0].message.content
    sandbox_result = run_sandbox(response_text)

    if sandbox_result["success"]:
        text = sandbox_result["stdout"]
        parts = text.split(",")
        chi2_value = parts[0].split("=")
        p_value = parts[1].split("=")

        calculated = {"chi2": float(chi2_value[1]), "p_value": float(p_value[1])}
    else:
        calculated = {"error": sandbox_result["error"]}

    new_log_entry =  {"agent": "modelling_agent", "action": f"outcome that modelling agent came with: {calculated}"}
    updated_log = state["decision_log"] + [new_log_entry]

    return {"calculated_data": calculated, "decision_log": updated_log, "modelling_counter": state["modelling_counter"]+1, "generated_code": response_text}

def route_after_modelling(state: ResearchState) -> str:
    calculated_data = state["calculated_data"]

    if "error" not in calculated_data:
        return "approved"
    elif state["modelling_counter"] < 3:
        return "needs_revision"
    else:
        return "failed"

#new node/agent
def interpretation_agent(state: ResearchState) -> dict:
    calculated= state["calculated_data"]   #read what modelling_agent produced, and what is in state
    
    prompt = f"""Here is a statistical result from a health survey. The following method was used to test whether physical activity (PhysActivity) is related to diabetes (Diabetes_binary) in this data:

{state['method_choice']}

Result:
Statistic: {calculated['chi2']}
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
builder.add_node("methods_agent", methods_agent)
builder.add_node("modelling_agent", modelling_agent)
builder.add_node("interpretation_agent", interpretation_agent)
builder.add_node("reviewer_agent", reviewer_agent)

#wire the steps
builder.add_edge(START, "data_agent")          
builder.add_edge("data_agent", "methods_agent")  
builder.add_edge("methods_agent", "modelling_agent")

builder.add_conditional_edges(
    "modelling_agent", route_after_modelling,
    {
        "approved": "interpretation_agent",
        "needs_revision": "modelling_agent",
        "failed": END
    }
)
 
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

graph.get_graph().draw_mermaid_png(output_file_path="graph.png") #create a graph to vizualise how the nodes interact with each other

result = graph.invoke({
    "given_data": "",
    "method_choice": "",
    "generated_code": "",
    "calculated_data": {},
    "interpretation": "",
    "reviewer_feedback": "",
    "reviewer_verdict": False,
    "revision_counter": 0,
    "modelling_counter": 0,
    "decision_log": [],
})

print(" --------------- THESE ARE PRINTED VALUES TO CHECK STATE --------------- ")

print(result["interpretation"])
print(result["method_choice"])
print(result["given_data"])
print(result["decision_log"])
print(result["calculated_data"])