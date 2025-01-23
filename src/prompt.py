from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder


system_prompt = (
"You are an expert Data Scientist assistat of qestion-answering tasks."
"Use the following pieces of retrieved context to answer "
"the question. If you don't find any related context then say that you "
"don't know. Do not give any halusinating answer of this. Use the three sentece maximum and keep the "
"answer concise."
"\n\n"
"{context}"
 )

chat_prompt = ChatPromptTemplate.from_messages([
  ("system", system_prompt),
  ("user", "{input}" )]
)


contextualize_system_prompt = (
  "Given a chat history and latest user question "
  "which might reference context in the chat history, "
  "formulates a standalone question which can be understood "
  "without the chat history. Do not answer the question, "
  "just reformulate it if needed otherwise retuen as it is"
)

contextualize_prompt = ChatPromptTemplate.from_messages([
  ("system", contextualize_system_prompt),
  MessagesPlaceholder(variable_name="chat_history"),
  ("user", "{input}"),
  ("system", "Context: {context}")
])