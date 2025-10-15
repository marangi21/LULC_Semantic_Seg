from terratorch.tasks import SemanticSegmentationTask

class DiffLRSemanticSegmentationTask(SemanticSegmentationTask):
    def configure_optimizers(self):
        params = [
            {"params": self.model.backbone.parameters(), "lr": 1e-6, "weight_decay": 1e-3},
        ]
        pass