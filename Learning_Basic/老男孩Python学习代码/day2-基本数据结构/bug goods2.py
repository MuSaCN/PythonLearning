# Author:Zhang Yuan

product_list=[
    ('Iphone',5800),
    ('Mac Pro',9800),
    ('Watch',10600),
    ('Coffee',30),
    ('Book',120)
]
shopping_list=[]
salary = input('Input your salary:')
#判断输入的字符串是否为数字
if salary.isdigit()==True:
    salary=int(salary)
    while True:
        #普通的索引方式
        '''
        for item in product_list:
            print(product_list.index(item),item)
        '''
        #enumerate相当于枚举式的索引，单独一个变量中存储“序号和内容”
        '''
        for item in enumerate(product_list):
            print(item)
        '''
        #两个变量分别存储：index-序号、item-内容
        for index,item in enumerate(product_list):
            print(index,item)
        user_choice=input('what do you want to buy?  or q?')
        if user_choice.isdigit()==True:
            user_choice=int(user_choice)
            #len()返回列表的长度
            if user_choice<len(product_list) and user_choice>=0:
                p_item=product_list[user_choice]
                if p_item[1]<=salary:
                    shopping_list.append(p_item)
                    salary-=p_item[1]
                    print('Added %s into shopping cart,your current balance is \033[31;1m%s\033[0m' %(p_item,salary))
                else:
                    print('\033[41;1mYour salary only have %s\033[0m'%salary)
            else:
                print('product code [%s] is not exist'%user_choice)
        elif user_choice=="q":
            print('------shopping list---')
            for p in shopping_list:
                print(p)
            print('your current balance:',salary)
            exit()
        else:
            print('invalid input')



