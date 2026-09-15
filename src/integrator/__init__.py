"""The Source Integrator: research, propose, approve, integrate.

A conversational agent that finds candidate sources for the catalog, works out
whether their licence permits use, ranks them, and — only after a person
approves — integrates them with full provenance.

The package is deliberately thin. The tools it calls live in `wisefood-mcp`,
the extraction pipelines it drives already exist in FoodScholar, and the
approval wall is a column in Postgres. What is here is the loop that joins
them: :mod:`integrator.agent`.
"""
