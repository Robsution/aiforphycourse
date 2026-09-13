kind = 'images'
if (kind == 'labels'):
    kind = kind + '-idx1-ubyte'
if (kind == 'images'):
    kind = kind + '-idx3-ubyte'
path = './data/validation/'
path = './mnist/'
f = open(f'{path}t10k-{kind}','rb')
somebytes = f.read(32)

print(f'{kind}')
print("Bytes:", somebytes)
print("Integers:", list(somebytes))
print("Hex:", somebytes.hex(" "))