# Author:Zhang Yuan

sentence=input("Sentence: ")
text_width=len(sentence)
box_width=text_width+6
left_margin=10
text_margin=(box_width-2-text_width)//2  # 整除,向下圆整

print()
print(" "*left_margin + "+" + "-"*(box_width-2) + "+")
print(" "*left_margin + "|" + " "*(box_width-2) + "|")
print(" "*left_margin + "|" + " "*text_margin+sentence+" "*text_margin + "|")
print(" "*left_margin + "|" + " "*(box_width-2) + "|")
print(" "*left_margin + "+" + "-"*(box_width-2) + "+")
print()






