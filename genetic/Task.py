"""Potwierdzam samodzielność powyższej pracy oraz niekorzystanie przeze mnie z niedozwolonych źródeł. ~Mateusz Bąkała"""


class Task:
    def __init__(self, target_func, genes_func, minimize=True):
        self.target = target_func
        self.genes = genes_func
        self.minimize = minimize

    def generate_genes(self):
        return self.genes()

    def evaluate(self, indv):
        indv.target = self.target(indv.genes)
