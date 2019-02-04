class ds:
    var1="hello"
    def __init__(self):
        print("Init!")
    def var2(self):
        print("world")
        return "world"


d=ds()
s=d.var2
print(s)