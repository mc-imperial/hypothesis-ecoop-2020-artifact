import ast


def size(lsi):
    return len(ast.literal_eval(lsi)[0])
