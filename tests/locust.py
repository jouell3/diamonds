import locust

class DiamondPredictionUser(locust.HttpUser):
    @locust.task
    def predict(self):
        self.client.post("/predict", json={
            "carat": 0.5,
            "cut": "Ideal",
            "color": "E",
            "clarity": "VS1",
            "depth": 61.5,
            "table": 55.0,
            "x": 5.0,
            "y": 5.0,
            "z": 3.0
        })
        
    @locust.task
    def predict_multiple(self):
        self.client.post("/predict_multiple", json=[
            {
                "carat": 0.5,
                "cut": "Ideal",
                "color": "E",
                "clarity": "VS1",
                "depth": 61.5,
                "table": 55.0,
                "x": 5.0,
                "y": 5.0,
                "z": 3.0
            },
            {
                "carat": 1.0,
                "cut": "Premium",
                "color": "D",
                "clarity": "VVS2",
                "depth": 62.0,
                "table": 57.0,
                "x": 6.5,
                "y": 6.5,
                "z": 4.0
            }
        ])

