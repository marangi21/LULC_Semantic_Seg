import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

class SegmentationLegend():
    """
    Una classe per gestire la creazione di colormap, normalizzazione e legende
    per la visualizzazione di maschere di segmentazione semantica.

    Incapsula la logica per garantire colori consistenti per ogni classe
    attraverso diverse visualizzazioni.
    """

    def __init__(self, class_mapping: dict, cmap_name: str = 'tab20'):
        if not class_mapping:
            raise ValueError("Il dizionario class_mapping non può essere vuoto o None.")
        
        # Ordinamento delle classi in base al loro ID originale per un ordine consistente
        class_items = sorted(class_mapping.items(), key=lambda x: int(x[1]))

        # Estrazione nomi delle classi nell'ordine corretto
        self.class_names = [item[0] for item in class_items]
        n_classes = len(self.class_names)

        # Generazione lista di colori basata sulla colormap scelta
        base_cmap = plt.get_cmap(cmap_name)
        # Usa il modulo per ciclare sui colori se le classi sono più dei colori base
        self.colors = [base_cmap(i % base_cmap.N) for i in range(n_classes)]

        # Creazione della colormap e della normalizzazione per i valori mappati (0, 1, 2...)
        self.cmap = ListedColormap(self.colors)
        self.norm = BoundaryNorm(np.arange(n_classes + 1) - 0.5, n_classes)

    def plot(self, bbox_to_anchor: tuple = (1, 1), loc: str = 'upper left'):
        """
        Crea e visualizza una legenda matplotlib basata sui dati della classe.
        Da chiamare dopo aver plottato la maschera.
        """
        patches = [
            mpatches.Patch(color=self.colors[i], label=f"{i}: {self.class_names[i]}")
            for i in range(len(self.class_names))
        ]
        plt.legend(
            handles=patches, 
            bbox_to_anchor=bbox_to_anchor, 
            loc=loc, 
            borderaxespad=0.
        )