# Author:Zhang Yuan

def lines(file):
    for line in file:
        yield line
    yield "\n"

#块输出
def blocks(file):
    block=[]
    for line in lines(file):
        #如果当前行strip后，有内容，则添加到列表
        if line.strip():
            #print(line)
            block.append(line)
        #如果当前行没有内容，且block[]存有内容，则打块输出
        elif block:
            #以""链接序列block的元素，生成新字符串，再strip
            yield "".join(block).strip()
            block=[]
