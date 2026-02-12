import asyncio
from llm_interpreter import SemanticInterpreter

async def test():
    interpreter = SemanticInterpreter()
    result = await interpreter.interpret("Give me something dark and driving")
    print(result)

asyncio.run(test())