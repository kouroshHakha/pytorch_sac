For running on singularity on RLL machines:
```
sudo singularity run --no-home --keep-privs --allow-setuid -B projects:/projects --writable  docker://kouroshhakha/osil:latest
```