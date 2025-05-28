class Clock:
    def __init__(self, hours, minutes, seconds):
        self.set_Clock(hours, minutes, seconds)

    def set_Clock(self, hours, minutes, seconds):
        if type(hours) == int and 0 <= hours < 24:
            self._hours = hours
        else:
            raise TypeError("Hours must be 0-23")
        if type(minutes) == int and 0 <= minutes < 60:
            self.__minutes = minutes
        else:
            raise TypeError("Minutes must be 0-59")
        if type(seconds) == int and 0 <= seconds < 60:
            self.__seconds = seconds
        else:
            raise TypeError("Seconds must be 0-59")

    def __str__(self):
        return f"{self._hours:02}:{self.__minutes:02}:{self.__seconds:02}"

    def tick(self):
        if self.__seconds == 59:
            self.__seconds = 0
            if self.__minutes == 59:
                self.__minutes = 0
                if self._hours == 23:
                    self._hours = 0
                else:
                    self._hours += 1
            else:
                self.__minutes += 1
        else:
            self.__seconds += 1


class Calendar:
    months = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    date_style = "British"

    @staticmethod
    def leapyear(year):
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    def __init__(self, d, m, y):
        self.set_Calendar(d, m, y)

    def set_Calendar(self, d, m, y):
        self.__days = d
        self.__months = m
        self.__years = y

    def __str__(self):
        if Calendar.date_style == "British":
            return f"{self.__days:02}/{self.__months:02}/{self.__years}"
        return f"{self.__months:02}/{self.__days:02}/{self.__years}"

    def advance(self):
        max_days = Calendar.months[self.__months - 1]
        if self.__months == 2 and Calendar.leapyear(self.__years):
            max_days += 1
        if self.__days == max_days:
            self.__days = 1
            if self.__months == 12:
                self.__months = 1
                self.__years += 1
            else:
                self.__months += 1
        else:
            self.__days += 1


class CalendarClock(Clock, Calendar):
    def __init__(self, d, m, y, h, min, s):
        Clock.__init__(self, h, min, s)
        Calendar.__init__(self, d, m, y)

    def tick(self):
        prev_hour = self._hours
        Clock.tick(self)
        if self._hours < prev_hour:
            self.advance()

    def __str__(self):
        return Calendar.__str__(self) + ", " + Clock.__str__(self)


x = CalendarClock(31, 12, 2013, 23, 59, 59)
print("One tick from", x, end=" → ")
x.tick()
print(x)
