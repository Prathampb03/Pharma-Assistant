from trulens.apps.langchain import TruChain
from trulens.core import TruSession
from graph import graph

# Initialize a TruLens session
session = TruSession()

# Instrument your LangChain application
tru_chain = TruChain(graph, app_name="graph")

# (Optional) Reset the database if needed
session.reset_database()
