# Project Notes

## Viva Explanation Points
1. **Why two inputs?** To detect change. Water looks like smooth/dark surfaces in SAR. By comparing before and after, we filter out permanent water bodies.
2. **Why infrastructure?** Roads act as drainage; buildings cause double-bounce. Knowing where they are helps the model understand complex reflection patterns.
3. **Why ResNet?** Proven feature extractor. Shared weights (Siamese) ensure we compare "apples to apples".

## Key Decisions
- **Dataset:** Sen1Floods11 (High quality hand-labeled subset).
- **Mode:** Fast (Bolivia/USA/Spain) for rapid iteration.
- **Hardware:** Mac Mini (M-series). Using MPS acceleration.
- **Precision:** FP32 (float32) because MPS has stability issues with FP16 for some ops.

## Known Issues
- GEE Auth requires Project ID with billing enabled.
- OSM rasterization needs to map correct pixel resolution (10m).
