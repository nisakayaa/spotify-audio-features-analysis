# Spotify Audio Features Analysis

Spotify'ın bir şarkıyı hangi özelliklerle "popüler" yaptığını anlamaya 
çalıştığım bir proje. Audio feature'larla (danceability, energy, valence vs.) 
popülerlik arasındaki ilişkiyi inceledim.

## Neden bu proje?
Çalma listemde son zamanlarda hep aynı tarzda şarkılar olduğunu fark ettim.
"Acaba Spotify benim için bir profil mi çıkarmış?" sorusundan başladı.

## Ne yaptım?
- 12,000 şarkının audio feature'larını çektim
- Popülerliği tahmin etmek için Random Forest eğittim
- Türleri K-Means ile kümelemeye çalıştım

## Bulgular
- Loudness ve energy, popülerlik için en güçlü iki özellik
- Random Forest'in R² skoru ~0.40 (bekledğimden düşük — popülerlik 
  audio'dan daha fazlasına bağlı)
- 2010'lardan sonra "loudness war" etkisi açıkça görülüyor

## Sınırlamalar
- Sentetik veri kullandım (Spotify API key'im yok)
- Popülerlik etiketi sosyal dinamikleri yansıtmıyor

## Kullanım
\`\`\`bash
pip install -r requirements.txt
python src/generate_data.py
python src/popularity_model.py
\`\`\`

## Author
Nisa Kaya — [github.com/nisakayaa](https://github.com/nisakayaa)
