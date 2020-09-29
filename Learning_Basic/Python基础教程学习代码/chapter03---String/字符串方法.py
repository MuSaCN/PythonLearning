# Author:Zhang Yuan

text="what is your name"
print(text.center(27,"*"))
a=["a","b","c","d","e"]
b="ABC"
print(b.join(a),b)

#f字符串
year = 2016
event = 'Referendum'
print(f'Results of the {year} {event}')

#字典传递到字符串
table = {'Sjoerd': 4127, 'Jack': 4098, 'Dcab': 8637678}
print('Jack: {0}; Sjoerd: {1}; '
      'Dcab: {2}'.format(table["Jack"],table["Sjoerd"],table["Dcab"]))
print('Jack: {0[Jack]}; Sjoerd: {0[Sjoerd]}; '
      'Dcab: {0[Dcab]}'.format(table))
table = {'Sjoerd': 4127, 'Jack': 4098, 'Dcab': 8637678}
print('Jack: {Jack:d}; Sjoerd: {Sjoerd:d}; Dcab: {Dcab:d}'.format(**table))

#在字符串输出时，{}里面有 ':' ，且后传递一个整数可以让该字段成为最小字符宽度。这在使列对齐时很有用。
for x in range(1, 11):
    print('{0:2d} {1:3d} {2:4d}'.format(x, x*x, x*x*x))

#字符串对象的 str.rjust() 方法通过在左侧填充空格来对给定宽度的字段中的字符串进行右对齐。类似的方法还有 str.ljust() 和 str.center()
for x in range(1, 11):
    print(str(x).rjust(1), str(x*x).rjust(3), end=' ')
    # Note use of 'end' on previous line
    print(repr(x*x*x).rjust(4))

print('12'.zfill(10))
print('-3.14'.zfill(10))
print('3.14159265359'.zfill(5))


