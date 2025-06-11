from fighting_robot import FightingRobot
from nursing_robot import NursingRobot


class FightingNurseRobot(NursingRobot, FightingRobot):

    def __init__(self, name, mode="nursing"):
        super().__init__(name)
        self.mode = mode

    def say_hi(self):
        if self.mode == "fighting":
            FightingRobot.say_hi(self)
        elif self.mode == "nursing":
            NursingRobot.say_hi(self)
        else:
            super().say_hi()
