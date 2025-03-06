import re
query = """
Check out a net with 6 faces below:
<image_0>Here are the steps to fold the net with face 3 as the base:
Step 1: Fold face 4 upwards
<image_1>
Step 2: Fold face 2 upwards
<image_2>
Step 3: Fold face 5 downwards towards face 2
<image_3>
Step 4: Fold face 6 upwards towards face 2
<image_4>
Based on the above steps, can the net be folded to form a cube, yes or no?
"""
matches = re.findall(r"<image_(\d+)>", query)

for match in matches:
    print(match)
    