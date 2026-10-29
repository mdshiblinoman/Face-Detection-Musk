# Face Mask Detection

এই project-এ local dataset দিয়ে PyTorch model train করে model file save করা হয় এবং Streamlit website-এ image upload করে prediction করা যায়।

## 1. Environment তৈরি

Windows:

```bash
py -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

সব platform-এ এরপর:

```bash
python -m pip install -r requirements.txt
```

## 2. Model train এবং save

Dataset-এর এই structure থাকতে হবে:

```text
data/
  with_mask/
  without_mask/
```

Training চালান:

```bash
python train_model.py --epochs 5
```

Trained model এখানে save হবে: `models/face_mask_model.pt`

এই command project folder-এর বাইরে অন্য folder থেকে চালালেও script নিজের folder-এর `data` এবং `models` path ব্যবহার করবে। নির্দিষ্ট dataset বা output দিতে চাইলে:

```bash
python train_model.py --data-dir "C:\path\to\data" --output "C:\path\to\model.pt"
```

CPU ধীর হলে `--epochs 3` ব্যবহার করা যায়। GPU থাকলে PyTorch সেটি স্বয়ংক্রিয়ভাবে ব্যবহার করবে।

## 3. Streamlit website চালানো

```bash
streamlit run app.py
```

Browser-এ দেখানো local URL খুলে ছবি upload করুন।

## 4. React Native mobile app

Mobile app-টি `mobile-app/` folder-এ আছে। এটি Expo দিয়ে Android/iOS-এ চলে এবং একই trained model-এর FastAPI backend ব্যবহার করে।

প্রথমে backend চালান:

```bash
python -m pip install -r requirements.txt
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

আপনার PC-এর local IP address দেখুন:

Windows:

```bash
ipconfig
```

Linux:

```bash
hostname -I
```

এরপর [mobile-app/App.js](./mobile-app/App.js)-এর `API_URL`-এ `http://YOUR_PC_IP:8000` লিখুন। ফোন এবং PC একই Wi-Fi network-এ থাকতে হবে।

Node.js install করে Expo app চালান:

```bash
cd mobile-app
npm install
npx expo start
```

Android ফোনে Expo Go app দিয়ে QR code scan করুন। API ঠিকমতো চলছে কি না দেখতে browser-এ `http://YOUR_PC_IP:8000/health` খুলে `{"status":"ok"}` দেখুন।

## Notebook

Colab-এর `/content` এবং Kaggle commands-এর বদলে local data path ব্যবহার করার জন্য `face-mask-detection.ipynb` চালাতে পারেন। Notebook-এর training cell শেষ হলে একই `models/face_mask_model.pt` তৈরি হবে।
