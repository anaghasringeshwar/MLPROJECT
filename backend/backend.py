from fastapi import FastAPI
import uvicorn
import recommend  # Import our recommendation model

app = FastAPI()

@app.get("/recommend/{image_id}")
async def get_recommendations(image_id: str):
    try:
        similar_images = recommend.find_similar(image_id)
        return {"image_id": image_id, "recommendations": similar_images}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
