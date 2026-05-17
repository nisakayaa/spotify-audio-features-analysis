# Spotify Audio Features Analysis

Spotify'ın bir şarkıyı popüler yapan ses özelliklerini anlamaya çalıştığım proje. Çalma listemi inceleyince fark ettim ki **belirli özellikler** sürekli karşıma çıkıyor — danceability yüksek, energy yüksek. Bu fark edişi bir veri sorusuna çevirdim: *"Bir şarkıyı 'hit' yapan ses özellikleri var mı, varsa hangileri?"*



## Sorduğum sorular

1. Popülerlikle en çok ilişkili ses özellikleri hangileri?
2. Pop müziğin "sesi" on yıllar içinde nasıl değişti?
3. Sadece ses özelliklerine bakarak türleri ayırt edebilir miyiz?
4. Şarkıları "duygu × enerji" düzleminde nereye konumlandırabiliriz?
5. Popülerliği tahmin eden bir model nasıl çalışırdı?

## Bulgular

- Popülerlikle en güçlü korelasyona sahip özellikler: **loudness**, **energy**, **danceability** (sırayla)
- 1990'lardan 2020'lere doğru **acousticness düşmüş**, **loudness artmış** (loudness war net görünüyor)
- K-Means (k=5) Hip-Hop ve EDM'i diğer türlerden başarıyla ayırıyor, ama Pop/Rock/R&B birbirine karışıyor — beklenen bir durum
- Mood quadrant'lara göre popüler şarkıların büyük çoğunluğu "**high energy + high valence**" (mutlu/heyecanlı) bölgesinde
- Random Forest ile popularity tahmininde **R² ~0.40** — düşük, çünkü popülerlik audio'dan çok fazlasına bağlı (pazarlama, sosyal medya, sanatçı tanınırlığı vs.)

## Yöntem

1. **Veri üretimi:** 12.000 şarkı, 8 tür profili, 1970-2024 yıl aralığı
2. **EDA:** Dağılımlar, korelasyon ısı haritası, on yıllık trendler
3. **Özellik mühendisliği:** Mood quadrant (energy × valence)
4. **Kümeleme:** K-Means (k=5), elbow method ile k seçimi
5. **Modelleme:** Random Forest Regressor, feature importance
6. **Yorumlama:** Hangi özellikler popülerliği gerçekten yordadı?

## Özellikler ne anlama geliyor?

| Özellik | Anlamı | Aralık |
|---------|--------|--------|
| Danceability | Şarkı dansa ne kadar uygun | 0-1 |
| Energy | Yoğunluk ve aktivite | 0-1 |
| Valence | Müzikal pozitiflik (mutlu vs hüzünlü) | 0-1 |
| Acousticness | Akustik mi elektronik mi | 0-1 |
| Speechiness | Konuşma içeriği yoğunluğu | 0-1 |
| Instrumentalness | Vokalsiz mi | 0-1 |
| Liveness | Canlı kayıt mı | 0-1 |
| Tempo | BPM | 50-200 |
| Loudness | dB cinsinden ses seviyesi | -60 - 0 |

## Kullandığım araçlar

- pandas, numpy
- scikit-learn (KMeans, RandomForestRegressor, StandardScaler)
- matplotlib, seaborn

## Çalıştırmak için

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/eda.py
python src/clustering.py
python src/popularity_model.py
```

## Sınırlamalar

Bu projedeki veri sentetik. Gerçek Spotify API'sinden çekmek için **Spotify Developer hesabı** ve `spotipy` kütüphanesi gerekiyor. Kod yapısı bunu destekliyor — sadece `generate_data.py` yerine Spotipy ile veri çeken bir modül eklemek yeterli:

```python
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())
features = sp.audio_features(track_ids)
```

## Not

Bu projede en çok eğlendiğim kısım **mood quadrant'lar** oldu. Bir şarkıyı energy ve valence ile 2D düzleme yerleştirip "Heyecanlı / Gergin / Sakin / Hüzünlü" kadranlarına ayırmak hem matematiksel hem de bana çok sezgisel geldi. Bazı çalma listelerimi gerçek hayatta da bu mantıkla ayırıyorum aslında.

Random Forest'in R² skorunun düşük çıkması başta hayal kırıklığıydı ama sonra fark ettim ki bu aslında **doğru sonuç**: popülerlik audio özelliklerinden çok daha fazlasına bağlı. Modelin "öğrenememesi" gerçek hayatın karmaşıklığını yansıtıyor, modelin kötülüğünü değil.

## Author

Nisa Kaya — [github.com/nisakayaa](https://github.com/nisakayaa)
