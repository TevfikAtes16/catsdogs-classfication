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

# Geliştirme için geniş CORS (prod'da domainlerini özel yazarsın)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = tf.keras.models.load_model(str(MODEL_PATH))
except Exception as e:
    raise RuntimeError(f"Model yüklenemedi: {MODEL_PATH}\n{e}")


def preprocess(img: Image.Image) -> np.ndarray:
    """
    test_savedmodel.py ile birebir: SADECE resize (normalize yok)
    """
    # EXIF'e göre döndür, RGB'ye çevir
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize((IMG, IMG))

    # float32 [0, 255] aralığında array
    arr = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(arr, 0)  # [1, H, W, 3]
    return x


@app.get("/")
def root():
    return {
        "message": "Cats & Dogs API up. Use POST /predict with form-data 'file'."
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    React Native tarafı:
        form.append("file", { uri, name: "upload.jpg", type: "image/jpeg" })
    şeklinde gönderiyor. Burada 'file' adıyla yakalıyoruz.
    """
    try:
        # Dosyayı oku
        raw = await file.read()
        if not raw:
            raise ValueError("Boş dosya alındı.")

        # Pillow ile aç
        img = Image.open(io.BytesIO(raw))
        x = preprocess(img)

        # Model çıktısı: tek örnek & tek nöron (sigmoid) varsayımı
        y = model.predict(x, verbose=0)
        # y: [1,1] veya [1] → skaler
        prob = float(np.squeeze(y))

        # Güvenli olsun: eğer [0,1] aralığı dışındaysa, logit kabul edip sigmoid uygula
        if prob < 0.0 or prob > 1.0:
            prob = float(tf.nn.sigmoid(prob))

        # 0.5 üzeri: dog, altı: cat
        label = "dog" if prob >= 0.5 else "cat"
        confidence = prob if label == "dog" else (1.0 - prob)

        # React Native PredictScreen şunu bekliyor:
        # result.class
        # result.confidence
        return {
            "class": label,
            "confidence": confidence,
        }

    except Exception as e:
        # Hata olursa 400 dön, RN tarafında Alert ile göreceksin
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Docker içinde 0.0.0.0'a bağlanıyoruz ki -p ile dışarı açabilelim
    uvicorn.run(app, host="0.0.0.0", port=8000)
