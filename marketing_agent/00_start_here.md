# Marketing Agent
This respository contains a walkthrough to make an agent using tools, Genie, and LangGraph.

It builds off the Genie [multiagent documentation](https://docs.databricks.com/aws/en/generative-ai/agent-framework/multi-agent-genie) and the [CME Marketing Campaign demo](https://www.databricks.com/resources/demos/tutorials/aibi-genie-marketing-campaign-effectiveness).

But it does a couple of extra things:
1. It builds a more realistic dataset using Batch Inference.
2. It then leverages that new dataset to build a creative ReAct agent with Vector Search. 
3. We then give the agent access to two tools - a simple UC function and a Genie Agent. 
4. We deploy this model and evaluate it using the Agent Evaluation framework.

## Running the repo
Follow the notebooks in the repo in order. 
All of the notebooks are designed to work with and tested on Serverless compute.

`01_data`: Loads the marketing dbdemo and runs batch inference to get the data ready.
`02_tools`: Sets up the SQL functions and vector search
`03_deploy`: Goes over the agent code and deploys it
`04_evaluate`: Tests our agent using the Mosaic Agent Evaluation framework

## Gotchas
There are two critical things you have to change when running the code:
- you will need to put your Genie Space ID in the config file
- you will need to change the campaigns table to the campaigns_fixed table in the Genie Space

You will also need a vector search endpoint. 
Remember to delete this at the end of the demo if you aren't planning on using it because it doesn't scale to zero!