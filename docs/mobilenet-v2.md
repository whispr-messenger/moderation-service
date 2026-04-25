# MobileNetV2-0.35

## Caractéristiques

- Taille : < 1 MB (TFLite)
- Latence : < 5ms sur mobile
- Framework : TensorFlow/Keras

## Pipeline

```
Dataset ──▶ Training ──▶ Export TFLite ──▶ Mobile
```

## Commandes

```bash
cd src/mobilenet_v2_small
python main.py --action train
python main.py --action eval
```
