"""Potwierdzam samodzielność powyższej pracy oraz niekorzystanie przeze mnie z niedozwolonych źródeł. ~Mateusz Bąkała"""
from sklearn.decomposition import PCA


class Task:
    def __init__(self, data, target):
        # pre-set PCA with 2 components, used for visualizations only
        self.pca = PCA(n_components=2)
        self.pca_data = None
        self.data = data
        self.target = target

    def get_x(self):
        """Returns data without target column (doesn't modify existing data!)."""
        return self.data.drop(self.target, axis=1)

    def generate_pca(self):
        """Trains PCA on X data."""
        self.pca_data = self.pca.fit_transform(self.get_x())
        return self
