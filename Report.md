#  CV-HW4 Stereo Matching

電機四 B04901190 范晟祐

## Part1

Let  $d = X_L -X_R$ , prove  $d = \frac{f \cdot b}{z}$

<img src="/Users/frank/Desktop/temp/1.png" style="zoom:80%"/>

Ans:
$$
b:X_L+b-X_R = z:z+f
$$

$$
b:d+b = z:z+f
$$

$$
dz+bz = bz+bf
$$

$$
d =\frac{b\cdot f}{z}
$$

## Part2

###Alogorithm

#### Cost Computation

Calculate the cost for each pixel with specific layer
$$
C_i,_j = (1-\alpha)\cdot min[\|I'_{i+l} - I_i\|,\tau_1]+\alpha \cdot min[\|\nabla_x I'_{i+l}- \nabla_xI_i|,\tau_2]
$$
####Cost Aggregation

1. Produce the cost volume, each layer has parameter $l$ , assigning the disparity from 1 to Maxdisp, and finsih it with above equation.
2. Apply an image guided filter on each layer of cost volume, using the input image as the reference of image.

####Disp. Optimization

Winner take all from cost volume layers, and choose the minimum value.

####Disp. Refinement

Apply a median weighted filter on the above labeled image.

###Results

####Tsukuba
<img src="/Users/frank/Desktop/temp/tsukuba.png" style="zoom:80%"/>
####Venus
<img src="/Users/frank/Desktop/temp/venus.png" style="zoom:80%"/>
####Teddy
<img src="/Users/frank/Desktop/temp/teddy.png" style="zoom:80%"/>
####Cones
<img src="/Users/frank/Desktop/temp/cones.png" style="zoom:80%"/>

###Bad Pixel Ratio

Tsukuba: 2.23%

Venus: 0.41%

Teddy: 9.63%

Cones: 7.40%

Average: 4.92%

###Reference

Fast Cost-Volume Filtering for Visual Correspondence and Beyond- Asmaa Hosni ; Christoph Rhemann ;

