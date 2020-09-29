# Author:Zhang Yuan
goods1=['G1',256]
goods2=['G2',300]
goods3=['G3',400]
goods4=['G4',123]
goods5=['G5',56]
goods6=['G6',321]

wallet=1000
ShoppingCart=[]
print('your wallet have $', wallet)
while (wallet > 56):
    buy=input("what do you want to buy? G1-G6 or Stop")
    if buy==goods1[0]:
        goods=goods1
        if wallet>goods[1]:
            ShoppingCart.append(goods1)
            wallet=wallet-goods[1]
            print("your shopping carts are:",ShoppingCart,' your wallet have',wallet)
        else:
            print("your need$", goods[1], 'but your have $', wallet)
    elif buy==goods2[0]:
        goods=goods2
        if wallet>goods[1]:
            ShoppingCart.append(goods2)
            wallet=wallet-goods[1]
            print("your shopping carts are:",ShoppingCart,' your wallet have',wallet)
        else:
            print("your need$", goods[1], 'but your have $', wallet)
    elif buy==goods3[0]:
        goods=goods3
        if wallet>goods[1]:
            ShoppingCart.append(goods3)
            wallet=wallet-goods[1]
            print("your shopping carts are:",ShoppingCart,' your wallet have',wallet)
        else:
            print("your need$", goods[1], 'but your have $', wallet)
    elif buy==goods4[0]:
        goods=goods4
        if wallet>goods[1]:
            ShoppingCart.append(goods4)
            wallet=wallet-goods[1]
            print("your shopping carts are:",ShoppingCart,' your wallet have',wallet)
        else:
            print("your need$", goods[1], 'but your have $', wallet)
    elif buy==goods5[0]:
        goods=goods5
        if wallet>goods[1]:
            ShoppingCart.append(goods5)
            wallet=wallet-goods[1]
            print("your shopping carts are:",ShoppingCart,' your wallet have',wallet)
        else:
            print("your need$", goods[1], 'but your have $', wallet)
    elif buy==goods6[0]:
        goods=goods6
        if wallet>goods[1]:
            ShoppingCart.append(goods6)
            wallet=wallet-goods[1]
            print("your shopping carts are:",ShoppingCart,' your wallet have',wallet)
        else:
            print("your need$",goods[1],'but your have $',wallet)
    elif buy=='Stop':
        print('your stop buy by yourself')
        print("your shopping carts are:", ShoppingCart, ' your wallet have', wallet)
        break
else:
    print("your have spend most of your money, your can't buy any more")


