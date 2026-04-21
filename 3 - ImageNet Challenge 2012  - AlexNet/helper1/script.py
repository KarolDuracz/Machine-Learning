a = 0
def test():
    global a
    for k in range(224):
        print((k * 4) + 11)
        if ((k * 4) + 11) > 224:
            break
        a += 1
 
test() 
print(a)