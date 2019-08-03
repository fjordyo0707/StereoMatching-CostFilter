# **StereoMatching-CostFilter**

## Contributor
Cheng-Yu Fan

## Dependencies
    python3
    opencv-python
    (Optional) Open3D
## Run
Run our Fast Cost-Volumne Filtering

```bash
python CostFilter.py -l <path to left image> -r <path to right image> -o <path to disparity image>
```
View the result

    python reconturction.py  -l <path to left image> -d  <path to disparity> -p <path to .ply file>
**Demo**

```bash
python CostFilter.py -l ./testdata/tsukuba/im3.png -r ./testdata/tsukuba/im4.png -o ./tsukuba.png
python recontruction.py  -l ./testdata/tsukuba/im3.png -d  tsukuba.png -p tsukuba.ply
```


## Results

![alt text](https://raw.githubusercontent.com/fjordyo0707/CGFinal-ImmerseIntheCanvas/master/img/1.png)
![alt text](https://raw.githubusercontent.com/fjordyo0707/CGFinal-ImmerseIntheCanvas/master/img/2.png)


## Reference
Fast Cost-Volume Filtering for Visual Correspondence and Beyond, Christoph Rhemann, Asmaa Hosni, Michael Bleyer, Carsten Rother, Margrit Gelautz, CVPR 2011

## Citation
If you use our code please cite the paper [C. Rhemann, A. Hosni, M. Bleyer, C. Rother, M. Gelautz, Fast Cost-Volume Filtering for Visual Correspondence and Beyond, CVPR11]

