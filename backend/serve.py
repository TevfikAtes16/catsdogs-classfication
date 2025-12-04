# serve.py
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # testteki gibi, sayısal tutarlılık için

from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import io
import numpy as np
import tensorflow as tf

IMG = 224

# ---- Modeli yükle (path'i sağlamlaştır) ----
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (BASE_DIR / "../service/catsdogs.keras").resolve()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # prod'da domainini yaz
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = tf.keras.models.load_model(str(MODEL_PATH))
except Exception as e:
    raise RuntimeError(f"Model yüklenemedi: {MODEL_PATH}\n{e}")

def preprocess(img: Image.Image):
    # test_savedmodel.py ile birebir: SADECE resize (normalize yok!)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize((IMG, IMG))
    arr = tf.keras.utils.img_to_array(img)  # float32 [0,255] aralığında
    x = np.expand_dims(arr, 0)              # [1, H, W, 3]
    return x

@app.get("/")
def root():
    return {"message": "Cats & Dogs API up. Use POST /predict with form-data 'file'."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw))
        x = preprocess(img)

        # testteki gibi tek örnek ve tek nöron/sigmoid kabulü
        y = model.predict(x, verbose=0)
        # y beklenen: shape [1,1] veya [1]; her durumda skaler çıkaralım
        prob = float(np.squeeze(y))  # logit değilse direkt p; logit ise aşağıda sigmoid uygula

        # Güvenli olsun: eğer değer [0,1] dışında ise bu logittir → sigmoid uygula
        if prob < 0.0 or prob > 1.0:
            prob = float(tf.nn.sigmoid(prob))

        label = "dog" if prob >= 0.5 else "cat"
        conf = prob if label == "dog" else (1 - prob)

        return {"class": label, "confidence": conf}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
