import ast


def size(ls):
    return len(ast.literal_eval(ls)[1])
