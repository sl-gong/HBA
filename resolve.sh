
mkdir -p /Users/gsl/work/slam/HBA/multi-strip/1lidar && \
for las in /Users/gsl/work/slam/HBA/multi-strip/1/LidarData/*.las; do \
  base=$(basename "$las" .las); \
  pos="/Users/gsl/work/slam/HBA/multi-strip/1/LidarData/${base}.pos"; \
  out="/Users/gsl/work/slam/HBA/multi-strip/1lidar/${base}.las"; \
  echo "Inverse Gauss(pos): ${base}.las"; \
  /Users/gsl/work/slam/HBA/bin/pcd_resolve \
    --mode inverse \
    --coord gauss \
    --pos "$pos" \
    --device /Users/gsl/work/slam/HBA/multi-strip/1/device_info.json \
    --las "$las" \
    --out "$out" \
    --use-point-time || exit 1; \
done

mkdir -p /Users/gsl/work/slam/HBA/multi-strip/1ned && \
for las in /Users/gsl/work/slam/HBA/multi-strip/1lidar/*.las; do \
  base=$(basename "$las" .las); \
  pos="/Users/gsl/work/slam/HBA/multi-strip/1/MVP-11-150.pos"; \
  out="/Users/gsl/work/slam/HBA/multi-strip/1ned/${base}.las"; \
  echo "forward ned(pos): ${base}.las"; \
  /Users/gsl/work/slam/HBA/bin/pcd_resolve \
    --mode forward \
    --coord ned \
    --pos "$pos" \
    --device /Users/gsl/work/slam/HBA/multi-strip/1/device_info.json \
    --las "$las" \
    --out "$out" \
    --use-point-time || exit 1; \
done

mkdir -p /Users/gsl/work/slam/HBA/multi-strip/1gas && \
for las in /Users/gsl/work/slam/HBA/multi-strip/1lidar/*.las; do \
  base=$(basename "$las" .las); \
  pos="/Users/gsl/work/slam/HBA/multi-strip/1/LidarData/${base}.pos"; \
  out="/Users/gsl/work/slam/HBA/multi-strip/1gas/${base}.las"; \
  echo "forward Gauss(pos): ${base}.las"; \
  /Users/gsl/work/slam/HBA/bin/pcd_resolve \
    --mode forward \
    --coord gauss \
    --pos "$pos" \
    --device /Users/gsl/work/slam/HBA/multi-strip/1/device_info.json \
    --las "$las" \
    --out "$out" \
    --use-point-time || exit 1; \
done