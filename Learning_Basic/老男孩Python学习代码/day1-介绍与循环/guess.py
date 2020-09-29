# Author:Zhang Yuan
#study
age_of_oldboy=57


count=0
while count<3:
    guess_age = int(input("guess age:"))
    if guess_age==age_of_oldboy:
        print("yes")
        break
    elif guess_age > age_of_oldboy:
        print("big")
    else:
        print("small")
    count+=1
else:
    print("you are tried too many");

for i in range(3):
    guess_age = int(input("guess age:"))
    if guess_age==56:
        print("yes")
        break
    elif guess_age > 56:
        print("big")
    else:
        print("small")
else:
    print("you are tried too many");