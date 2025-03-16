category_prompt_template = """
        Please act as a robust and well-trained intent classifier that can identify the most 
        likely category that the questions refers to, WITHOUT USING the proper and common noun subject from the user's query. 
        It can be ΚΑΡΤΕΣ (cards), ΚΑΤΑΝΑΛΩΤΙΚΑ (personal loans) and ΣΤΕΓΑΣΤΙΚΑ (mortgage) or ΑΛΛΟ (all the rest).
        ΑΛΛΟ is the category for the queries that do not be classified in the other categories.
        The identified category must be only one word and one of the ΚΑΡΤΕΣ, ΚΑΤΑΝΑΛΩΤΙΚΑ, ΣΤΕΓΑΣΤΙΚΑ or ΑΛΛΟ.

        User's query: {question}

        Category:
        """