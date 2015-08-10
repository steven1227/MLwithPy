from operator import add
expr = "28+32+++32++39"
reduce(add, map(int, filter(bool, expr.split("+"))))

print reduce(add, map(int, filter(bool, expr.split("+"))))