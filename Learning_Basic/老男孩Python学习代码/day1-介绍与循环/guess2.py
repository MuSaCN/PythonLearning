# Author:Zhang Yuan
age_of_oldboy=56
count=0
keep=1

while count<3 and keep==1:
    guess_age = int(input("guess age:"))
    if guess_age==age_of_oldboy:
        print("yes")
        break
    elif guess_age > age_of_oldboy:
        print("big")
    else:
        print("small")
    count+=1
    if count>=3:
        keep = int(input("keep:"));
        print(keep)
        if keep==1:
            count=0
else:
    print("Finish Game");
