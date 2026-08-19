with open("scripts/ai/python_worker.py", "r") as f:
    code = f.read()

code = code.replace("server.bind(('127.0.0.1', port))", "server.bind(('0.0.0.0', port))")

with open("scripts/ai/python_worker.py", "w") as f:
    f.write(code)
