from locust import HttpUser, task, between

class FitVerseUser(HttpUser):
    wait_time = between(1, 5)  # Wait 1-5 seconds between tasks
    
    @task
    def predict(self):
        self.client.post("/predict", json={
            "Weight": 70,
            "Height": 1.75,
            "BMI": 22.86,
            "Age": 30,
            "Gender": "Male"
        })
