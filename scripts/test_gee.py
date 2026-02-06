import ee
import google.auth

PROJECT_ID = 'rare-ridge-486516-k1'

def test_init():
    try:
        print(f"Testing Standard Init with project {PROJECT_ID}...")
        ee.Initialize(project=PROJECT_ID)
        print("Success!")
        return
    except Exception as e:
        print(f"Standard Init failed: {e}")

    try:
        print("Testing ADC Init...")
        credentials, project = google.auth.default(
            scopes=['https://www.googleapis.com/auth/earthengine', 'https://www.googleapis.com/auth/cloud-platform']
        )
        ee.Initialize(credentials=credentials, project=PROJECT_ID)
        print("Success with ADC!")
    except Exception as e:
        print(f"ADC Init failed: {e}")

if __name__ == "__main__":
    test_init()
