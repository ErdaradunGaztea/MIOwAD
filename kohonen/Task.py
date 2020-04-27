"""Potwierdzam samodzielność powyższej pracy oraz niekorzystanie przeze mnie z niedozwolonych źródeł. ~Mateusz Bąkała"""


class Task:
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def get_x(self):
        """Returns data without target column (doesn't modify existing data!)."""
        return self.data.drop(self.target, axis=1)
