<!-- https://github.com/AngshumanChakraborty/RAG_LLM_Langchain_ChromaDB.git -->

# Required tools
Operating System- Linux (tested on Ubuntu distro) / Windows with the following installed:  
Nvidia CUDA Driver (tested on version 12.2) installed  
Current Docker version installed  
Python 3.10


# Docker container setup
Configured in Dockerfile  
Further configuration in devcontainer.json


# Vscode IDE
Import the project directory and it will detect the .devcontainer directory and it will download the required container image and configure it by installing all the packages specified in the requirements.txt. The installation will create a virtual environment to install all packages. Post installation, check the if the python interpreter being used is from the virtual environment using the below commands:

python
> import os, sys  
> os.path.dirname(sys.executable)

If the interpretor is not the same as virtual environment, source it using the below command:  

> source /workspaces/interviewVscodeEnv/bin/activate  

Also select the python interpreter in the VSCode IDE. Press the Shift + Ctrl + P keys and in the command palette, type in "python" from where select the Select Interpreter option and the choose the python interpreter within the virtual environment.

# How to run the RAG bot
Open rag_implementation.py file with VSCode and hit the Run button or type the following command in the VSCode IDE terminal:

python3 /workspaces/mercInterview/src/rag_implementation.py  
or  
python3 /workspaces/mercInterview/src/rag_implementation.py --data_path "data/text" --file_type ".txt" --vectorstore_path "./sql_chroma_db" --model_id "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  
or  
put custom values

# Tools and models used

LLM used- TinyLlama/TinyLlama-1.1B-Chat-v1.  
Vector Database- Chroma

# Estimated time spent

Decision to use which model, download it to use locally, create the project structure along with docker image, determine the requirements.txt, git, .env, source the text data- approx 5-6 hours  
To implement the actual code flow- approx 4-5 hours  
Time to address couple of serious bugs- approx 3 hours

