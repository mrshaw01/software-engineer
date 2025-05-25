"""Robot class with class attribute to store Asimov's Three Laws of Robotics."""


class Robot:
    Three_Laws = (
        "A robot may not injure a human being or, through inaction, allow a human being to come to harm.",
        "A robot must obey the orders given to it by human beings, except where such orders would conflict with the First Law.",
        "A robot must protect its own existence as long as such protection does not conflict with the First or Second Law.",
    )


for i, law in enumerate(Robot.Three_Laws, 1):
    print(f"{i}: {law}")
