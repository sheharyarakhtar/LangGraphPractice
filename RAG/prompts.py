from langchain.prompts import ChatPromptTemplate, PromptTemplate

rag_prompt = ChatPromptTemplate.from_messages(
	    [
	        ("system", """
	        	You are an AI assistant designed to provide accurate and relevant answers based on the provided context.
                Use the information from the context as your sole source of knowledge.
                Only answer questions if the information is available in the context. 
                Do not attempt to answer questions outside of the provided context.
	        	"""),
            ("system",""""
                 ================================================================================
                    Context: {context}
             
            """),
	        ("system","""
                ================================================================================
	            	Previous conversation:\n
	            	{history}
                ================================================================================
            """),
	        ("human", "{question}"),
            ("{human_input}")
	    ]
	)


search_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a keyword extractor. 
        The user will send in their question. 
        Instead of doing a similarity search on the question itself, extract keywords from the question.
        You can add relevant key-words that are not mentioned in the question as well.
        example: What are some good dog breeds?
        keywords: [dogs, canines, german shepherds] etc.
        respond with no more than 3 keywords so that searching is quicker. Be efficient.
        Your output must be in JSON format, with a list of keywords.
    """),
        ("Question: {q}")
    ])