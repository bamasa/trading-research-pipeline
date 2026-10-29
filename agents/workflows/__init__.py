"""The research loop's orchestrator: the tool policy, the prompts and the loop.

``policy`` decides every tool call an agent makes; ``prompts`` assembles each
role's system prompt and step prompt and reads its handback; ``research_loop``
runs the steps in order, checks the gates and keeps the run's record. Nothing
here is study code, and nothing under ``src/`` imports it.
"""
